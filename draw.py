#!/bin/python

import argparse
import math
import numpy
from numpy.polynomial import polynomial as poly

class Canvas(object):
    
    class Color:
        VIOLET = "\033[38;5;92m"
        GREEN = "\033[38;5;34m"
        ENDC = "\033[0m"
        
    # this will allow to store different shapes in the same grid
    class CellValue:
        ORIGIN = -1
        BLANK = 0
        
        def getChar(val):
            pass
        
    def fromArgs(args):
       blank = args['blank']
       fill = args['fill']
       r = 20 #default canvas
       xLimit = numpy.array(args['x_limit'])
       yLimit = numpy.array(args['y_limit'])
       zLimit = numpy.array(args['z_limit'])
       grid = Canvas(fill, blank, numpy.sort(numpy.array([xLimit, yLimit, zLimit])))
       return grid
    
    def __init__(self, fill, blank, interval):
        self.fill = fill
        self.blank = blank
        self.separator = '-'
        self.shapes = []
        """
        the window is an array of intervals, one for each dimension in use R(3, 2)
        """
        self.setWindow(interval)
        
    def setWindow(self, interval):
        self.window = interval
        # add 1 to interval ends since range() is not inclusive
        self.window[:, 1] += 1        
        dimension = (
            abs(interval[0, 1] - interval[0, 0]),
            abs(interval[1, 1] - interval[1, 0]),
            abs(interval[2, 1] - interval[2, 0]),
        )
        self.data = numpy.ndarray(dimension)
        self.redraw()

    def redraw(self):
        for s in self.shapes:
            s.draw(self)
    
    def set(self, point, value=1):
        if (point > self.window[:,0]).all() and (point < self.window[:,1]).all():
            self.data[(point - self.window[:,0])] = value
          
    def get(self, point):
        if (point > self.window[:,0]).all() and (point < self.window[:,1]).all():
            return self.data.item(*(point - self.window[:,0]))
  
    def setgrid(self, mask, value=1):
        #set mask
        self.data[mask] = value

    def grid(self):
        xx = numpy.arange(*self.window[0,:])
        yy = numpy.arange(*self.window[1,:])
        zz = numpy.arange(*self.window[2,:])
        return numpy.meshgrid(yy, xx, zz)
    
    def printp(self, value, color=''):
        print(color + value + Canvas.Color.ENDC, end=' ')
        
    def endl(self):
        print('\n', end='')

    def zSeparator(self, length):
        for i in range(0, length):
            print(self.separator, end=' ')
        self.endl()
    
    def show(self):
        for z in range(0, self.data.shape[2]):
            for y in range(0, self.data.shape[0]):
                for x in range(0, self.data.shape[1]):
                    if (self.data.item(*(numpy.array([y, x, z]))) == 1):
                        self.printp(self.fill, Canvas.Color.GREEN)
                    else:
                        self.printp(self.blank)
                self.endl()
            self.zSeparator(self.data.shape[1])
    
    def addShape(self, shape):
        self.shapes.append(shape)
        shape.draw(self)

class Shape(object):
    
    def __init__(self):
        self.color = None

    def setColor(self, color):
        self.color = color;

    def draw(self, canvas):
        # abstract
        pass
        
    def fromArgs(args):
        #static builder method
        pass

class Circle(Shape):
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        super(Circle, self).__init__()
        
    def draw(self, canvas):
        gx, gy, gz = canvas.grid()
        g = numpy.array([gx, gy, gz])
        g = g.transpose([1, 2, 3, 0])
        g -= self.center
        norm = numpy.linalg.norm(g, axis=3)
        circle = numpy.around(norm - self.radius) == 0
        #print circle
        canvas.setgrid(circle)

    def fromArgs(args):
        r = args['radius']
        center = numpy.array([0, 0, 0])
        circle = Circle(center, r)
        return circle

class Ellipse(Shape):
    
    def __init__(self, r_x, r_y, r_z, center):
        self.rX = r_x
        self.rY = r_y
        self.rZ = r_z
        self.center = center
        # this is empirical
        self.threshold = 0.65

    def draw(self, canvas):
        gx, gy, gz = canvas.grid()
        c = numpy.zeros((3,3,3))
        c[2,0,0] = 1/(self.rX**2)
        c[0,2,0] = 1/(self.rY**2)
        c[0,0,2] = 1/(self.rZ**2)
        # todo add z
        c[0,0,0] = ((self.center[0]/self.rX)**2 + 
                    (self.center[1]/self.rY)**2 + 
                    (self.center[2]/self.rZ)**2 - 1)
        c[1,0,0] = 2*self.center[0]/self.rX**2
        c[0,1,0] = 2*self.center[1]/self.rY**2
        c[0,0,1] = 2*self.center[2]/self.rZ**2
        points = numpy.polynomial.polynomial.polyval3d(gx, gy, gz, c)
        candidates = numpy.around(points, decimals=3)
        ellipse = (candidates >= 0) & (candidates <= self.threshold)
        canvas.setgrid(ellipse)
    
    def fromArgs(args):
        rX = args['r_x']
        rY = args['r_y']
        rZ = args['r_z']
        center = numpy.array(args['center'])
        ellipse = Ellipse(rX, rY, rZ, center)
        return ellipse

class Plane(Shape):
    
    def __init__(self, normal):
        self.normal = normal
    
    def draw(self, canvas):
        gx, gy, gz = canvas.grid()
        c = numpy.zeros((2,2,2))
        c[1,0,0] = self.normal[0]
        c[0,1,0] = self.normal[1]
        c[0,0,1] = self.normal[2]
        points = poly.polyval3d(gx, gy, gz, c)
        plane = numpy.around(points) == 0
        canvas.setgrid(plane)
        
    def fromArgs(args):
        normal = args['normal']
        plane = Plane(normal)
        return plane

class Thorus(Shape):
    
    def __init__(self, r_thorus, r_pipe, center):
        self.radius = r_thorus
        self.pipe_radius = r_pipe
        self.center = center
        
    def draw(self, canvas):
        gx, gy, gz = canvas.grid()
        c = numpy.zeros((4,4,4))
        c[1,0,0] = self.normal[0]
        c[0,1,0] = self.normal[1]
        c[0,0,1] = self.normal[2]
        points = poly.polyval3d(gx, gy, gz, c)
        plane = numpy.around(points) == 0
        canvas.setgrid(plane)
        
    def fromArgs(args):
        radius = args['radius']
        pipe_radius = args['pipe_radius']
        center = args['center']
        return Thorus(radius, pipe_radius, center)

def main():
    parser = argparse.ArgumentParser(description="generate a discrete circle map")
    parser.add_argument('-b', '--blank', type=str, default=' ', dest='blank', required=False, help='Char to use for blank cell')
    parser.add_argument('-f', '--fill', type=str, default='+', dest='fill', required=False, help='Char to use for block cell')
    parser.add_argument('-x', '--x-limit', nargs=2, type=int, default=[-20, 20], dest='x_limit', 
                        required=False, help='X axis limits (min, max)')
    parser.add_argument('-y', '--y-limit', nargs=2, type=int, default=[-20, 20], dest='y_limit', 
                        required=False, help='Y axis limits (min, max)')
    parser.add_argument('-z', '--z-limit', nargs=2, type=int, default=[0, 0], dest='z_limit', 
                        required=False, help='Z axis limits (min, max)')
    subp = parser.add_subparsers(help="sub-command --help");

    # create parser for the circle command
    parser_c = subp.add_parser('circle', help="circle --help")
    parser_c.add_argument('-r', '--radius', type=int, default=5, dest='radius', required=False, help='Radius of the circle, default %(default)s')
    parser_c.set_defaults(func=Circle.fromArgs)
    
    # create parser for the ellipse command
    parser_e = subp.add_parser('ellipse', help="ellipse --help")
    parser_e.add_argument('-m', '--radius-x', type=int, default=5, dest='r_x', required=False, 
                          help='X radius of the ellipse, default %(default)s')
    parser_e.add_argument('-n', '--radius-y', type=int, default=5, dest='r_y', required=False, 
                          help='Y radius of the ellipse, default %(default)s')
    parser_e.add_argument('-o', '--radius-z', type=int, default=5, dest='r_z', required=False, 
                          help='Z radius of the ellipse, default %(default)s')
    parser_e.add_argument('-c', '--center', type=int, nargs=3, default=[0,0,0], dest='center', required=False,
                          help='Center of the ellipse, default %(default)s')
    parser_e.set_defaults(func=Ellipse.fromArgs)
    
    # create parser for the plane command
    parser_p = subp.add_parser('plane', help="plane --help")
    parser_p.add_argument('-n', '--normal', type=int, nargs=3, default=[1,0,0], dest='normal', required=False,
                          help='Normal to the plane, default %(default)s')
    parser_p.set_defaults(func=Plane.fromArgs)
    
        # create parser for the thorus command
    parser_t = subp.add_parser('thorus', help="thorus --help")
    parser_t.add_argument('-r', '--radius', type=int, default=5, dest='radius', required=False,
                          help='Radius of the thorus, from center to the middle of the pipe, default %(default)s')
    parser_t.add_argument('-p', '--pipe', type=int, default=2, dest='pipe_radius', required=False,
                          help='Radius of the pipe, default %(default)s')
    parser_t.add_argument('-c', '--center', type=int, nargs=3, default=[0,0,0], dest='center', required=False,
                          help='Center of the thorus, default %(default)s')
    parser_t.set_defaults(func=Thorus.fromArgs)

    # parse the arguments
    args = vars(parser.parse_args())
    canvas = Canvas.fromArgs(args)
    if ('func' in args):
        shape = args['func'](args)
        canvas.addShape(shape)
        canvas.show()
    else:
        parser.print_help()
    
if __name__ == "__main__":
    main()
