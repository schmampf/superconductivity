# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:19:36 2018

@author: David
"""

import numpy as np
from vapory import *
io.POVRAY_BINARY = r"C:\Program Files\POV-Ray\v3.7\bin\pvengine.exe"




class Round_Cylinder(POVRayElement):
    """ Todo"""

class Object(POVRayElement):
    """ Todo"""
    
class SphereSweep(POVRayElement):
    """ Todo"""
    
    
wire_radius = 0.75
wire = SphereSweep(
          "b_spline", "12",
          [0,-30,-29], wire_radius,
          [0,-30,-25], wire_radius,
          [0,-7,-23], wire_radius,
          [0,0,-17], wire_radius,
          [0,0,-11], wire_radius,
          [0,0,-9], wire_radius,

          [0,0,9], wire_radius,
          [0,0,11], wire_radius,
          [0,0,17], wire_radius,
          [0,-7,23], wire_radius,
          [0,-30,25], wire_radius,
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
          
          
#wire = SphereSweep(
#          "b_spline", "15",
#          [0,-25,-35], wire_radius,
#          [2,-20,-30], wire_radius,
#          [1,-10,-28], wire_radius,
#          [1,-7,-23], wire_radius,
#          [-1,-4,-21], wire_radius,
#          [0,-4,-18], wire_radius,
#          [0,0,-15], wire_radius,
#          [0,0,-9], wire_radius,
#          
#          
#          [0,0,9], wire_radius,
#          [0,0,13], wire_radius,
#          [-3,-1,15], wire_radius,
#          [0,-3,16], wire_radius,
#          [+4,-9,23], wire_radius,
#          [0,-28,25], wire_radius,
#          [0,-30,29], wire_radius,
#
#           Texture( 
#            Pigment( 
#              'color', 'OrangeRed' 
#            ),
#            Finish(  
#            #'phong', 0.9,
#            'ambient', 0.1, 
#            'diffuse', 0.5,
#            'metallic', 0.8,
#            'reflection', 0.2,)
#            ),
##            Normal(
##                  'bumps', 0.3,
##                  'scale', 0.2,
##                  ),
#            'rotate', [0,0,0],
#            'translate', [0,30,-3],
#            'scale', [.8, .8, .8],
#          )
#          




#          [0,-25,-35], wire_radius,
#          [2,-20,-30], wire_radius,
#          [1,-10,-28], wire_radius,
#          [1,-7,-23], wire_radius,
#          [-1,-4,-21], wire_radius,
#          [0,-4,-18], wire_radius,
#          [0,0,-15], wire_radius,
#          [0,0,-9], wire_radius,
#          
#
#          
#          [0,0,9], wire_radius,
#          [0,0,11], wire_radius,
#          [0,0,13], wire_radius,
#          [-3,-3,17], wire_radius,
#          [0,-7,16], wire_radius,
#          [+4,-9,23], wire_radius,
#          [0,-28,25], wire_radius,
#          [0,-30,29], wire_radius,