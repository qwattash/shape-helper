#!/bin/python

import argparse
import time
import numpy
from numpy.polynomial import polynomial as poly
from scipy.signal import fftconvolve, convolve
from scipy.ndimage.morphology import binary_dilation

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

    # ------------------------------------- Dataset access
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
  
    def setgrid(self, mask, value=1, downsample=False):
        #set mask
        if (downsample):
            #the mask given is too large, it must be
            # downsampled to fit the output data size
            src_shape = mask.shape
            dst_shape = self.data.shape
            ci = numpy.ceil(src_shape[0]/dst_shape[0])
            cj = numpy.ceil(src_shape[1]/dst_shape[1])
            ck = numpy.ceil(src_shape[2]/dst_shape[2])
            """
            premessa:
            le parti commentate vengono dall'implementazione usando un mean 
            filter, le parti attuali vengono da un'altra 
            implementazione più semplice che
            fa uso di una cosa che si chiama morfologia
            come al solito è un parolone per dire una cosa semplice D:
            l'idea di fondo è questa:
            io ho la matrice input con alcuni punti a 1 e il resto a zero
            i punti a 1 sono i punti sul luogo geometrico e voglio "propagarli"
            nell'output.
            qeul che faccio è prendere i punti della matrice input ogni 
            ci sulla x, cj sulla y e ck sulla z, questo mi permette di
            estrarre il numero di punti esattamente uguale a quelli
            necessari per riempire la matrice di output
            esempio in meno-d:
            se ho due matrici una 4x4 e una 2x2 e voglio passare dalla 
            4x4 alla 2x2 avrò tipo:
            [1< 2 3< 4,         [? ?,
             5 6 7 8,          ? ?] 
             9< 1 2< 3,
             4 5 6 7]
            ora qui ci e cj sono 2 perchè il fattore di scala è 2 ok
            quindi prendo i punti ogni 2 cioè got it!
            in 3D è uguale solo meno intuitivo questo l'ho capito
            ottimo, ora il problema è questo
            [0 1 0 1
             1 1(1) 1(1) 1(1),          ? ?] 
             0 1(1) 0<lui(1) 1(1), se c'è almeno un elemento della matrice
             1 1(1) 1(1) 1(1)]  a 1 su cui ho sovrapposto un elemento a 1 di struct allora mette il punto centrale a 1 ok in pratica in questo modo prendo i
            pezzi mancanti cioè mi guardo intorno, è simile ad un mean filter
            però siccome del numero non me ne frega faccio questa cosa binaria
            che è più veloce scusa brb un sec kk 
            eccomi, si comunque intuitivamente ci sono.
            in pratica mi viene 0 ma il luogo geometrico è ben presente in 
            quella zona capito.. o uno cambia il 
            "fattore di ridimensionamento"..
            l'operatore fa questa cosa:
            in pratica prende la matrice di input e per ogni punto ci
            sovrappone l'elemento "struct" di cui sotto che in questo caso è
            tutto a 1 ed è di dimensione variabile, qui assumi che sia 3x3 e
            il centro venga sovrapposto ad ogni punto
            cioè
            struct =
            [1 1 1,
             1 1< 1, lui è il centro e viene "sovrapposto"
             1 1 1]
            """
            #kernel = numpy.ones((ci,cj,ck)) * 1/(ci*cj*ck)
            struct_dimension = numpy.array([ci,cj,ck])
            # make the structuring element with odd dimension
            struct_dimension = numpy.add(struct_dimension, (struct_dimension + 1) % 2)
            struct = numpy.ones(struct_dimension, dtype=bool)
            #mask[:] *= (ci*cj*ck)
            #kernel = numpy.array([[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]]) * 1/27
            # N-d convolution with mean filter
            #mean = fftconvolve(mask, kernel)
            #print(numpy.count_nonzero(mean.flatten()))
            #print(mask.size)
            (si, sj, sk) = numpy.array(numpy.meshgrid(numpy.arange(0, mask.shape[0], ci, dtype=numpy.int16),
                                                      numpy.arange(0, mask.shape[1], cj, dtype=numpy.int16),
                                                      numpy.arange(0, mask.shape[2], ck, dtype=numpy.int16), 
                                                      sparse=True))
            origin = numpy.floor(struct_dimension / 2).astype(int)
            # the mask for dilation saves time by reducing the number of points
            # evaluated from (k*n)^3 to n^3 with n = dim(self.data)
            dilation_mask = numpy.zeros(src_shape, dtype=bool)
            dilation_mask[si,sj,sk] = True
            mean = binary_dilation(mask, struct, origin=origin, border_value=0, mask=dilation_mask)
            self.data = mean[si,sj,sk] != 0
            #self.show(dbg=True)
            return
        self.data[mask] = value

    def grid(self, samples=1, sparse=False):
        xx = numpy.arange(self.window[0,0], 
                          self.window[0,1], 
                          1/samples)
        yy = numpy.arange(self.window[1,0], 
                          self.window[1,1], 
                          1/samples)
        zz = numpy.arange(self.window[2,0], 
                          self.window[2,1], 
                          1/samples)
        return numpy.meshgrid(yy, xx, zz, sparse=sparse)

    # --------------------- OUTPUT FUNCTIONS
    def printp(self, value, color=''):
        # point print
        print(color + value + Canvas.Color.ENDC, end=' ')

    def printd(self, value, color=''):
        # debug print
        print("[{0:04}]".format(value), end=' ')
        
    def endl(self):
        #endline print
        print('\n', end='')

    def zSeparator(self, length):
        for i in range(0, length):
            print(self.separator, end=' ')
        self.endl()
    
    def show(self, dbg=False):
        for z in range(0, self.data.shape[2]):
            for y in range(0, self.data.shape[0]):
                for x in range(0, self.data.shape[1]):
                    if dbg:
                        self.printd(self.data.item(*(numpy.array([y, x, z]))))
                    elif (self.data.item(*(numpy.array([y, x, z]))) == 1):
                        self.printp(self.fill, Canvas.Color.GREEN)
                    else:
                        self.printp(self.blank)
                self.endl()
            self.zSeparator(self.data.shape[1])
            
    # --------------------- Shape handling functions
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
        R = self.radius
        r = self.pipe_radius
        # coefficents of the thorus polynomial
        # the polynomial is derived with basic arithmetic from
        # (R - sqrt(x^2 + y^2))^2 + z^2 = r^2
        # R = thorus radius
        # r = pipe radius
        self.c = numpy.zeros((5,5,5))
        self.c[4,0,0] = 1 #x^4
        self.c[0,4,0] = 1 #y^4
        self.c[0,0,4] = 1 #z^4
        self.c[2,0,0] = -2*(r^2 + R^2) #x^2
        self.c[0,2,0] = -2*(r^2 + R^2) #y^2
        self.c[0,0,2] = 2*(R^2 - r^2) #z^2
        self.c[2,2,0] = 2 #x^2y^2 
        self.c[2,0,2] = 2 #x^2z^2
        self.c[0,2,2] = 2 #y^2z^2
        self.c[0,0,0] = -2*(r*R)^2 + r^4 + R^4
        
    def draw(self, canvas):
        gx, gy, gz = canvas.grid(samples=10, sparse=False)
        start = time.time()
        points = poly.polyval3d(gx, gy, gz, self.c) 
        thorus = numpy.around(points) == 0
        print("--- %s seconds ---" % (time.time() - start,))
        canvas.setgrid(thorus, downsample=True)
        
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
