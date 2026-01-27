# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:20:36 2018

@author: David

"""

print('test')

import numpy as np
from vapory import *
import io
io.POVRAY_BINARY = r"C:\Program Files\POV-Ray\v3.7\bin\pvengine.exe"
#C:\Program Files\POV-Ray\v3.7




class Round_Cylinder(POVRayElement):
    """ Todo"""

class Object(POVRayElement):
    """ Todo"""
    
class SphereSweep(POVRayElement):
    """ Todo"""
    

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
    
 