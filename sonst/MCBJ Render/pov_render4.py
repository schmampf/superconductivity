# -*- coding: utf-8 -*-
"""
Created on Wed May 02 09:19:01 2018

@author: David Weber
"""

import scipy, io
import numpy as np
from vapory import *
io.POVRAY_BINARY = r"C:\Program Files\POV-Ray\v3.7\bin\pvengine.exe"




class Round_Cylinder(POVRayElement):
    """ Todo"""

class Object(POVRayElement):
    """ Todo"""
    
class SphereSweep(POVRayElement):
    """ Todo"""

zoom = 2.2


for config in [8]:
#for config in [1,2,3,5,6,7,8,9,10,11,12,13]:
    

    # load coordinates to render
    coord = np.loadtxt(r"config_%i.txt"%config, delimiter=",")   # config 3 single-atom
    # 4 broken; 5 single-atom stretched; 6,7,8,9 double, 12 dimer

    z_offset = -50
    camera = Camera(
                    'location', [60*zoom, 15, 25*zoom-40],
                    'look_at', [0, 15, z_offset],
                    "right", "x*image_width/image_height/2.",
#                    "focal_point", [8, 0, 8],
#                    "aperture", 3*zoom,
#                    "blur_samples", 200,
#                    "confidence", 1.,
#                    "variance", 1/1024.
                    )
    
    
    
    lights = [ 
                LightSource( [60, 30, -15], 'color', 'Grey' ),
                #LightSource( [60, 30, 0], 'color', 'Grey' ),
                LightSource( [60, 30, 15], 'color', 'Grey' ),
    
            ]
    
    
    plane = Plane([0,1,0], 1, 
                      Texture( 
                        Pigment( 
                          'color', 'Red'#'OrangeRed'
                        ),
                        Finish(  
                        'phong', 0.8,
                        'ambient', 0.2, 
                        'diffuse', 0.1),
                        ),
                        Normal(
                        'bumps', 8,
                        'scale', 1,
                        ),
                      'translate', [0,-50,0] )
    
    
    fog = Fog( "distance", 8000,
              "color", "Grey",#"Gold",
              "fog_type", 2,
              "fog_offset", 25,
              "fog_alt", 1,)
    
    background = Background("color", "Grey")
    
    
    
    a = np.linalg.norm(coord[0]-coord[1])
    a = 6.75/np.sqrt(3)
    
    
    
    alpha = np.deg2rad(45)
    
    
    # defining objects
    object_list = []
    
    coord *= 2.
    a *= 2.
    
    coord = list(coord)
    #a = 4.2
    angle = 0.5
    offset_z = -.4
    offset_z *= a
    spacing = .8 * a
    for z in np.arange(6, 110):#110
      for x in np.arange(-z, z, 1):  
        for y in np.arange(-z, z, 1):
    
          xnew, ynew, znew = (x + (z%2)/2.)*a, (y + (z%2)/2.)*a,  z*a/2.
          
    
          d = np.sqrt(xnew**2+ynew**2)
          phi = np.arctan2(xnew, ynew)
          phi += alpha
          
          xnew = np.sin(phi)*d
          ynew = np.cos(phi)*d  
          
          flattening = 1.
          if z > 9:
              flattening += (z-6)/800.
              
          
          if abs(d) / (abs(znew)**flattening+10) < angle and (abs(d)+20) / (abs(znew**flattening)+4) > angle: 
          
            xnew += np.random.rand()-0.5
            ynew += np.random.rand()-0.5
            if z < 15:
                coord.append(np.array([xnew,ynew,znew + offset_z + spacing, 1]))
            coord.append(np.array([xnew,ynew,-znew + offset_z - spacing, -1]))
    
      
      
    for idx, (x,y,z,info) in enumerate(coord):
      if info == 0:
        color = "Silver"
      if info == 1 or info == -1:
        color = "Silver"
      if info == 2:
        color = "Blue"
      
      sphere = Sphere( [x,y,z], a/2.5, 
                      Texture( 
                        Pigment( 
                          'color', color
                        ),
    
                        Finish(  
                        'metallic', 0.8,
                        'reflection', 0.5,
                        'diffuse', 0.5,
                        'phong', 0.2,
                        'ambient', 0.0,
                        )
                      )
                    )
                
      object_list.append(sphere)
      
    import battery
    import wire
    
#    sphere = Sphere( [0,0,0], 5, 
#                      Texture( 
#                        Pigment( 
#                          'color', "Blue"
#                        ),
#                      )
#                    )
#                
#    object_list.append(sphere)
    
      
    object_list.extend(lights)
    object_list.append(plane)
    #object_list.append(battery.battery)
    #object_list.append(wire.wire)
    object_list.append(fog)
    object_list.append(background)
    
    
    scene = Scene(camera = camera , # a Camera object
                  objects = object_list, # POV-Ray objects (items, lights)
                  included = ["colors.inc", "shapes.inc"]) # headers that POV-Ray may need
    
    
    filename = 'configuration.png'
    full_name = "%03.3f_%i_%s"%(zoom, config, filename)
    
    #scene.render('configuration_hq.png', width=9000, height=6000)
#    scene.render(full_name, width=6000, height=4000)
    scene.render(full_name, width=1200, height=800)
    import time
    time.sleep(0.1)
    
    
    import gc
    gc.collect()