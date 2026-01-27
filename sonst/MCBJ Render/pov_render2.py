# -*- coding: utf-8 -*-
"""
Created on Wed May 02 09:19:01 2018

@author: David Weber
"""

import scipy
import numpy as np
from vapory import *
io.POVRAY_BINARY = r"C:\Program Files\POV-Ray\v3.7\bin\pvengine.exe"




class Round_Cylinder(POVRayElement):
    """ Todo"""

class Object(POVRayElement):
    """ Todo"""
    
class SphereSweep(POVRayElement):
    """ Todo"""

# load coordinates to render
#coord = scipy.loadtxt(r"config_1.txt", delimiter=",")
coord = scipy.loadtxt(r"config_2.txt", delimiter=",")

camera = Camera( 'location', [61, 15, 10], 
                'look_at', [0, 0, 0],
                "right", "x*image_width/image_height",
                "focal_point", [0, 0, 0],
                "aperture", 4,
                "blur_samples", 250,
                "confidence", 1.,
                "variance", 1/1024.
                )

#        focal_point <0.20,1.5,-5.25>
#        aperture 0.7     // 0.05 ~ 1.5
#        blur_samples 100 // 4 ~ 100
#        confidence 0.9   // 0 ~ 1
#        variance 1/128   // 1/64 ~ 1/1024 ~

lights = [ 
            LightSource( [40, 40, 0], 'color', 'Grey' ),
            #LightSource( [40, 40, -50], 'color', 'White' ),
            LightSource( [40, 40, 50], 'color', 'White' ),
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
                  'translate', [0,-35,0] )


 
battery = Union( Object(                  
 "Round_Cylinder", "(", [0,0,-10], [0,0,1], 5.001, .5, 0, ")",

   Texture( 
    Pigment( 
      'color', 'Blue' 
    ),
    Finish(  
    'phong', 0.5,
    'ambient', 0.1, 
    'diffuse', 0.5,
    'metallic', 0.4,
    'reflection', 0.15,)
  )),
     
 Object(                 
 "Round_Cylinder", "(", [0,0,-1], [0,0,10], 5, .5, 0, ")",

   Texture( 
    Pigment( 
      'color', 'Red' 
    ),
    Finish(  
    'phong', 0.5,
    'ambient', 0.1, 
    'diffuse', 0.5,
    'metallic', 0.4,
    'reflection', 0.15,
    )
  )

),
    Object(                 
 "Round_Cylinder", "(", [0,0,9], [0,0,11], 1.5, .3, 0, ")",

   Texture( 
    Pigment( 
      'color', 'Silver' 
    ),
    Finish(  
    'phong', 0.5,
    'ambient', 0.1, 
    'diffuse', 0.5,
    'metallic', 0.4,
    'reflection', 0.3,)
  )

),
      'rotate', [0,0,0],
      'translate', [0,30,-3],
      'scale', [.8, .8, .8],
)
    
    
wire_radius = 0.6
wire = SphereSweep(
          "b_spline", "16",
          [0,-25,-35], wire_radius,
          [2,-20,-30], wire_radius,
          [1,-10,-28], wire_radius,
          [1,-7,-23], wire_radius,
          [-1,-4,-21], wire_radius,
          [0,-4,-18], wire_radius,
          [0,0,-15], wire_radius,
          [0,0,-9], wire_radius,
          
#          [0,-30,-29], wire_radius,
#          [0,-30,-25], wire_radius,
#          [0,-7,-23], wire_radius,
#          [0,0,-17], wire_radius,
#          [0,0,-11], wire_radius,
#          [0,0,-9], wire_radius,
          
          [0,0,9], wire_radius,
          [0,0,11], wire_radius,
          [0,0,13], wire_radius,
          [-3,-3,17], wire_radius,
          [0,-7,16], wire_radius,
          [+4,-9,23], wire_radius,
          [0,-28,25], wire_radius,
          [0,-30,29], wire_radius,

           Texture( 
            Pigment( 
              'color', 'OrangeRed' 
            ),
            Finish(  
            #'phong', 0.9,
            'ambient', 0.1, 
            'diffuse', 0.5,
            'metallic', 0.8,
            'reflection', 0.2,)
            ),
#            Normal(
#                  'bumps', 0.3,
#                  'scale', 0.2,
#                  ),
            'rotate', [0,0,0],
            'translate', [0,30,-3],
            'scale', [.8, .8, .8],
          )
          

    
battery2 = Union( Object(                  
 "Round_Cylinder", "(", [0,0,-5], [0,0,-3], 8, .3, 0, ")",

   Texture( 
    Pigment( 
      'color', 'Silver' 
    ),
    Finish(  
    'phong', 0.5,
    'ambient', 0.1, 
    'diffuse', 0.5,
    'metallic', 0.4,)
  )),
     
 Object(                 
 "Round_Cylinder", "(", [0,0,0], [0,0,2], 4, .3, 0, ")",

   Texture( 
    Pigment( 
      'color', 'Silver' 
    ),
    Finish(  
    'phong', 0.5,
    'ambient', 0.1, 
    'diffuse', 0.5,
    'metallic', 0.4,)
  )

),


      'rotate', [0,0,0],
      'translate', [0,28,0],
      'scale', [.8, .8, .8],
)    






    
    

fog = Fog( "distance", 4000,
          "color", "Grey",#"Gold",
          "fog_type", 2,
          "fog_offset", 25,
          "fog_alt", 1,)

background = Background("color", "Grey")



a = np.linalg.norm(coord[0]-coord[1])
a = 6.95/np.sqrt(3)
a = 6.75/np.sqrt(3)
alpha = np.deg2rad(45)


# defining objects
object_list = []

coord = list(coord)
#a = 4.2
angle = 0.5
offset_z = -.6
offset_z *= a
for z in np.arange(6, 30):
  for x in np.arange(-2*z, 2*z, 1):  
    for y in np.arange(-2*z, 2*z, 1):

      xnew, ynew, znew = (x + (z%2)/2.)*a, (y + (z%2)/2.)*a,  z*a/2.
      

      d = np.sqrt(xnew**2+ynew**2)
      phi = np.arctan2(xnew, ynew)
      phi += alpha
      
      xnew = np.sin(phi)*d
      ynew = np.cos(phi)*d      
      
      if abs(d) / (abs(znew**1.05)+3) < angle and not (abs(d)+15) / (abs(znew)+3) < angle: #and d < 4+abs(z)/4.  and d < 6+abs(z)/6.
      
        xnew += np.random.rand()-0.5
        ynew += np.random.rand()-0.5
        coord.append(np.array([xnew,ynew,znew + offset_z, 1]))
        coord.append(np.array([xnew,ynew,-znew + offset_z, -1]))

  
  
  
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
#                    Finish(  
#                    'phong', 0.5,
#                    'ambient', 0.8, 
#                    'diffuse', 0.3,
#                    'reflection',0.4,
#                    'metallic', 0.8,
#                    )
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
  
object_list.extend(lights)
object_list.append(plane)
object_list.append(battery)
object_list.append(wire)
object_list.append(fog)
object_list.append(background)


scene = Scene(camera = camera , # a Camera object
              objects = object_list, # POV-Ray objects (items, lights)
              included = ["colors.inc", "shapes.inc"]) # headers that POV-Ray may need

# passing 'ipython' as argument at the end of an IPython Notebook cell
# will display the picture in the IPython notebook.
scene.render('configuration.png', width=3000, height=2000)
#scene.render('configuration.png', width=1200, height=800)
import time
time.sleep(0.1)

#from IPython.display import Image
#Image(filename=r"D:\Desktop\configuration.png") 

