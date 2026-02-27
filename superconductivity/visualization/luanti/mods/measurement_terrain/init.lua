local modname = minetest.get_current_modname()
local modpath = minetest.get_modpath(modname)
local worldpath = minetest.get_worldpath()

-- ---- CONFIG (adjust these)
local RAW_FILE = worldpath .. "/map.u16le"
local META_FILE = worldpath .. "/map.json"

-- ---- Load metadata (NX/NY)
local mf = assert(io.open(META_FILE, "r"))
local meta = minetest.parse_json(mf:read("*a"))
mf:close()

local NX = assert(meta.nx, "meta.nx missing")
local NY = assert(meta.ny, "meta.ny missing")

-- ---- Coloring contract (palette + height scaling)
local HMIN = meta.hmin
local HMAX = meta.hmax

-- Optional separate limits for colormap scaling ("contrast knob").
-- If not provided, we fall back to the terrain export range.
-- These should be heights in the same units as the heightmap values `h`.
local CMAP_HMIN = meta.cmap_hmin or HMIN
local CMAP_HMAX = meta.cmap_hmax or HMAX

-- Palette texture must be resolvable as a TEXTURE NAME (not a filesystem path).
-- It can be provided by a global mod, a worldmod, or a texture pack.
local PALETTE_TEX = meta.palette or "cmap.png"

minetest.log("action", ("[measurement_terrain] palette texture = %s"):format(PALETTE_TEX))
minetest.log(
  "action",
  ("[measurement_terrain] cmap limits (heights): cmap_hmin=%s cmap_hmax=%s (export hmin=%s hmax=%s)")
    :format(tostring(CMAP_HMIN), tostring(CMAP_HMAX), tostring(HMIN), tostring(HMAX))
)

-- Dedicated surface node that will be tinted via param2 palette index
local SURF_NODE = modname .. ":surface"

local function clamp(x, a, b)
  if x < a then return a end
  if x > b then return b end
  return x
end

local function height_to_color(h)
  -- Map height h in [CMAP_HMIN,CMAP_HMAX] to palette index 0..255
  if not CMAP_HMIN or not CMAP_HMAX or (CMAP_HMAX == CMAP_HMIN) then
    return 0
  end
  local t = (h - CMAP_HMIN) / (CMAP_HMAX - CMAP_HMIN)
  t = clamp(t, 0.0, 1.0)
  -- round to nearest integer
  return math.floor(t * 255 + 0.5)
end

minetest.register_node(SURF_NODE, {
  description = "Measurement Surface",
  -- Use a very bright base texture so palette colors are not darkened by stone.
  -- `default_snow.png` exists in Minetest Game; if you use a different game,
  -- swap this to an available white texture (e.g. `wool_white.png`).
  tiles = {"default_snow.png"},
  paramtype2 = "color",
  palette = PALETTE_TEX,
  -- Prevent accidental edits (digging). This is a visualization world.
  diggable = false,
  drop = "",
  can_dig = function(pos, player)
    return false
  end,
  groups = {unbreakable = 1, not_in_creative_inventory = 1},
  sunlight_propagates = false,
  is_ground_content = true,
})

-- ---- Load raw heightmap once (uint16 little-endian row-major)
local f = assert(io.open(RAW_FILE, "rb"))
local raw = f:read("*a")
f:close()

-- Validate file size
local expected = NX * NY * 2
assert(#raw == expected,
  ("map.u16le size mismatch: got %d expected %d (NX=%d NY=%d)")
  :format(#raw, expected, NX, NY)
)

local function get_h(ix, iz)
  if ix < 0 or ix >= NX or iz < 0 or iz >= NY then
    return nil
  end
  local idx = (iz * NX + ix) * 2 + 1
  local lo = raw:byte(idx)
  local hi = raw:byte(idx + 1)
  return lo + hi * 256
end

minetest.log("action", ("[measurement] NX=%d NY=%d raw=%d"):format(NX, NY, #raw))
minetest.log("action", ("[measurement] h(0,0)=%s h(NX-1,NY-1)=%s")
  :format(get_h(0,0), get_h(NX-1,NY-1)))

-- world placement in x/z
local ORIGIN_X = 0
local ORIGIN_Z = 0

-- nodes per pixel (1 = 1:1)
local SCALE_X = 1
local SCALE_Z = 1

-- vertical placement
local BASE_Y = 300

-- player settings
local WALK_SPEED = 2.0  -- 1.0 is default
local SPAWN_Y_OFFSET = 20  -- spawn above terrain (nodes)
local AUTO_GRANT_ALL = true

-- cinematic preset A: fixed daylight "survey" look
local FIX_TIMEOFDAY = true
local TIMEOFDAY = 0.62  -- ~late afternoon
local TIME_FREEZE_PERIOD_S = 2.0

-- ---- Node IDs (Minecraft-like defaults)
local c_air, c_stone, c_surf, c_under

local function grant_all_privs(player)
  if not AUTO_GRANT_ALL then
    return
  end

  local name = player:get_player_name()
  if not name or name == "" then
    return
  end

  local privs = minetest.get_player_privs(name) or {}
  for pname, _ in pairs(minetest.registered_privileges) do
    privs[pname] = true
  end

  minetest.set_player_privs(name, privs)
end

local function resolve_node(name, fallback)
  if minetest.registered_nodes[name] then
    return name
  end
  if fallback and minetest.registered_nodes[fallback] then
    return fallback
  end
  return nil
end

minetest.register_on_mods_loaded(function()
  c_air = minetest.get_content_id("air")

  local stone = resolve_node("default:stone", nil)
  local under = resolve_node("default:stone", nil)

  assert(stone and under, "Required default nodes not registered")

  c_stone = minetest.get_content_id(stone)
  c_under = minetest.get_content_id(under)
  c_surf = minetest.get_content_id(SURF_NODE)
end)

local function get_spawn_pos()
  -- Center of the heightmap in world coordinates
  local sx = ORIGIN_X + math.floor((NX * SCALE_X) / 2)
  local sz = ORIGIN_Z + math.floor((NY * SCALE_Z) / 2)

  local ix = math.floor((sx - ORIGIN_X) / SCALE_X)
  local iz = math.floor((sz - ORIGIN_Z) / SCALE_Z)
  local h = get_h(ix, iz) or 0

  -- Spawn well above the surface to avoid collision before the chunk is generated.
  -- We will still "snap" to a safe air position once the area is emerged.
  local margin = math.max(SPAWN_Y_OFFSET, 40)

  return {
    x = sx,
    y = BASE_Y + h + margin,
    z = sz,
  }
end

local function ensure_safe_spawn(player)
  -- Ensure the player ends up in air above the surface.
  -- Works even if the area is not generated yet by emerging the target chunk.
  local p = get_spawn_pos()

  local radius = 2
  local emerge_min = {
    x = p.x - radius,
    y = p.y - 80,
    z = p.z - radius,
  }
  local emerge_max = {
    x = p.x + radius,
    y = p.y + 120,
    z = p.z + radius,
  }

  local function try_place()
    -- If node data is not available yet, keep waiting.
    local n = minetest.get_node_or_nil(p)
    if not n then
      return false
    end

    -- Move upward until we find air. Cap search to avoid infinite loops.
    local y = p.y
    for _ = 1, 200 do
      local nn = minetest.get_node_or_nil({ x = p.x, y = y, z = p.z })
      if nn and nn.name == "air" then
        -- Put player here. Add a tiny extra offset to avoid edge collisions.
        player:set_pos({ x = p.x + 0.5, y = y + 0.5, z = p.z + 0.5 })
        return true
      end
      y = y + 1
    end

    -- Fallback: force a high-altitude spawn.
    player:set_pos({ x = p.x + 0.5, y = p.y + 100, z = p.z + 0.5 })
    return true
  end

  -- Trigger map generation/loading around the target.
  minetest.emerge_area(emerge_min, emerge_max, function(_, _, remaining)
    if remaining == 0 then
      -- Defer placement by a tick to allow VoxelManip writes to land.
      minetest.after(0.1, function()
        if not player or not player:is_player() then
          return
        end
        try_place()
      end)
    end
  end)

  -- Also attempt an immediate placement (if area is already loaded).
  minetest.after(0, function()
    if not player or not player:is_player() then
      return
    end
    try_place()
  end)
end

local function apply_player_settings(player)
  -- Increase movement speed without requiring the "fast" privilege.
  -- Note: this is a physics override and may be clamped by some games/mods.
  player:set_physics_override({ speed = WALK_SPEED })

  -- Disable clouds for a clear view of the terrain.
  -- The fields are supported in modern Luanti builds; if a field is ignored,
  -- Luanti will fall back gracefully.
  if player.set_clouds then
    player:set_clouds({ density = 0 })
  end

  -- Reduce / disable fog and set a clear sky.
  if player.set_sky then
    -- "regular" keeps the default sky rendering; fog settings below will
    -- further reduce haze. Some games override sky settings.
    player:set_sky({
      type = "regular",
      clouds = false,
      fog = {
        fog_start = 1.0,
        fog_end = 0.0,
        fog_color = "#000000",
      },
    })
  end

  if player.set_fog then
    player:set_fog({
      fog_start = 1.0,
      fog_end = 0.0,
    })
  end

  -- Hide HUD elements for a clean, cinematic view.
  -- This removes the wielded-item arm, crosshair, and the hotbar.
  if player.hud_set_flags then
    player:hud_set_flags({
      wielditem = false,
      crosshair = false,
      hotbar = false,
      healthbar = false,
      breathbar = false,
      minimap = false,
      minimap_radar = false,
    })
  end
end

minetest.register_on_newplayer(function(player)
  grant_all_privs(player)
  apply_player_settings(player)
  ensure_safe_spawn(player)
end)

minetest.register_on_joinplayer(function(player)
  grant_all_privs(player)

  if not _G._measurement_privs_logged then
    _G._measurement_privs_logged = true
    local name = player:get_player_name()
    local p = minetest.get_player_privs(name) or {}
    minetest.log("action", ("[measurement_terrain] privs for %s: fly=%s fast=%s noclip=%s")
      :format(name, tostring(p.fly), tostring(p.fast), tostring(p.noclip)))
  end

  apply_player_settings(player)

  -- Safety: if the current position is inside a solid node, respawn safely.
  local pos = player:get_pos()
  local nn = minetest.get_node_or_nil(pos)
  if nn and nn.name ~= "air" then
    ensure_safe_spawn(player)
    return
  end

  -- If the player spawns outside the dataset area, move them onto it.
  local inside_x = (pos.x >= ORIGIN_X) and (pos.x < ORIGIN_X + NX * SCALE_X)
  local inside_z = (pos.z >= ORIGIN_Z) and (pos.z < ORIGIN_Z + NY * SCALE_Z)
  if not (inside_x and inside_z) then
    ensure_safe_spawn(player)
  end
end)

-- Keep lighting stable for a cinematic "survey" look.
if FIX_TIMEOFDAY then
  local acc = 0.0
  minetest.register_globalstep(function(dtime)
    acc = acc + dtime
    if acc >= TIME_FREEZE_PERIOD_S then
      acc = 0.0
      minetest.set_timeofday(TIMEOFDAY)
    end
  end)
end

minetest.register_on_generated(function(minp, maxp, seed)
  -- If the chunk is completely outside the dataset rectangle, erase it.
  local outside_x = (maxp.x < ORIGIN_X) or (minp.x >= ORIGIN_X + NX * SCALE_X)
  local outside_z = (maxp.z < ORIGIN_Z) or (minp.z >= ORIGIN_Z + NY * SCALE_Z)

  local vm, emin, emax = minetest.get_mapgen_object("voxelmanip")
  local area = VoxelArea:new({ MinEdge = emin, MaxEdge = emax })
  local data = vm:get_data()
  local param2 = vm:get_param2_data()

  -- if outside_x or outside_z then
  --   -- Fill entire chunk volume with air.
  --   for z = minp.z, maxp.z do
  --     for y = minp.y, maxp.y do
  --       local vi = area:index(minp.x, y, z)
  --       for x = minp.x, maxp.x do
  --         data[vi] = c_air
  --         vi = vi + 1
  --       end
  --     end
  --   end

  --   vm:set_data(data)
  --   vm:calc_lighting()
  --   vm:write_to_map()
  --   return
  -- end

  for z = minp.z, maxp.z do
    local iz = math.floor((z - ORIGIN_Z) / SCALE_Z)
    for x = minp.x, maxp.x do
      local ix = math.floor((x - ORIGIN_X) / SCALE_X)
      local h = get_h(ix, iz)
      if h then
        local top = BASE_Y + h

        for y = minp.y, maxp.y do
          local vi = area:index(x, y, z)
          if y <= top then
            data[vi] = c_surf
            -- the column share the same color.
            -- param2[vi] = height_to_color(h) 
            -- Color by height above BASE_Y: all blocks at the same (y-BASE_Y)
            param2[vi] = height_to_color(y - BASE_Y)
          else
            data[vi] = c_air
          end
        end
      end
    end
  end

  vm:set_data(data)
  vm:set_param2_data(param2)
  vm:calc_lighting()
  vm:write_to_map()
end)