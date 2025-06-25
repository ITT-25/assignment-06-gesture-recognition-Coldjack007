import recognizer
from recognizer import Point
import pyglet
from pyglet.window import mouse
import math

WINDOW_HEIGHT = 400
WINDOW_WIDTH = 420

is_down = False
points = []

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

dollar = recognizer.DollarRecognizer()

batch = pyglet.graphics.Batch()

class User_input:
    def __init__(self, x, y):
        self.Bubble = pyglet.shapes.Circle(x, y, 2.0, color=(255,0,0), batch=batch)

class Unistroke_input:
    def __init__(self, x, y):
        self.Bubble = pyglet.shapes.Circle(x, y, 2.0, color=(0,0,255), batch=batch)
        
array_user_input = []
array_uni_input = [#[Point(78,149),Point(78,153), Point(78,157), Point(78,160), Point(79,162), Point(79,164), Point(79,167), Point(79,169), Point(79,173), Point(79,178), Point(79,183), Point(80,189), Point(80,193), Point(80,198), Point(80,202), Point(81,208), Point(81,210), Point(81,216), Point(82,222), Point(82,224), Point(82,227), Point(83,229), Point(83,231), Point(85,230), Point(88,232), Point(90,233), Point(92,232), Point(94,233), Point(99,232), Point(102,233), Point(106,233), Point(109,234), Point(117,235), Point(123,236), Point(126,236), Point(135,237), Point(142,238), Point(145,238), Point(152,238), Point(154,239), Point(165,238), Point(174,237), Point(179,236), Point(186,235), Point(191,235), Point(195,233), Point(197,233), Point(200,233), Point(201,235), Point(201,233), Point(199,231), Point(198,226), Point(198,220), Point(196,207), Point(195,195), Point(195,181), Point(195,173), Point(195,163), Point(194,155), Point(192,145), Point(192,143), Point(192,138), Point(191,135), Point(191,133), Point(191,130), Point(190,128), Point(188,129), Point(186,129), Point(181,132), Point(173,131), Point(162,131), Point(151,132), Point(149,132), Point(138,132), Point(136,132), Point(122,131), Point(120,131), Point(109,130), Point(107,130), Point(90,132), Point(81,133), Point(76,133)], 
#[Point(127,141),Point(124,140), Point(120,139), Point(118,139), Point(116,139), Point(111,140), Point(109,141), Point(104,144), Point(100,147), Point(96,152), Point(93,157), Point(90,163), Point(87,169), Point(85,175), Point(83,181), Point(82,190), Point(82,195), Point(83,200), Point(84,205), Point(88,213), Point(91,216), Point(96,219), Point(103,222), Point(108,224), Point(111,224), Point(120,224), Point(133,223), Point(142,222), Point(152,218), Point(160,214), Point(167,210), Point(173,204), Point(178,198), Point(179,196), Point(182,188), Point(182,177), Point(178,167), Point(170,150), Point(163,138), Point(152,130), Point(143,129), Point(140,131), Point(129,136), Point(126,139)]
#[Point(91,185),Point(93,185), Point(95,185), Point(97,185), Point(100,188), Point(102,189), Point(104,190), Point(106,193), Point(108,195), Point(110,198), Point(112,201), Point(114,204), Point(115,207), Point(117,210), Point(118,212), Point(120,214), Point(121,217), Point(122,219), Point(123,222), Point(124,224), Point(126,226), Point(127,229), Point(129,231), Point(130,233), Point(129,231), Point(129,228), Point(129,226), Point(129,224), Point(129,221), Point(129,218), Point(129,212), Point(129,208), Point(130,198), Point(132,189), Point(134,182), Point(137,173), Point(143,164), Point(147,157), Point(151,151), Point(155,144), Point(161,137), Point(165,131), Point(171,122), Point(174,118), Point(176,114), Point(177,112), Point(177,114), Point(175,116), Point(173,118)], 
#[Point(123,129),Point(123,131), Point(124,133), Point(125,136), Point(127,140), Point(129,142), Point(133,148), Point(137,154), Point(143,158), Point(145,161), Point(148,164), Point(153,170), Point(158,176), Point(160,178), Point(164,183), Point(168,188), Point(171,191), Point(175,196), Point(178,200), Point(180,202), Point(181,205), Point(184,208), Point(186,210), Point(187,213), Point(188,215), Point(186,212), Point(183,211), Point(177,208), Point(169,206), Point(162,205), Point(154,207), Point(145,209), Point(137,210), Point(129,214), Point(122,217), Point(118,218), Point(111,221), Point(109,222), Point(110,219), Point(112,217), Point(118,209), Point(120,207), Point(128,196), Point(135,187), Point(138,183), Point(148,167), Point(157,153), Point(163,145), Point(165,142), Point(172,133), Point(177,127), Point(179,127), Point(180,125)], 
#[Point(81,219),Point(84,218), Point(86,220), Point(88,220), Point(90,220), Point(92,219), Point(95,220), Point(97,219), Point(99,220), Point(102,218), Point(105,217), Point(107,216), Point(110,216), Point(113,214), Point(116,212), Point(118,210), Point(121,208), Point(124,205), Point(126,202), Point(129,199), Point(132,196), Point(136,191), Point(139,187), Point(142,182), Point(144,179), Point(146,174), Point(148,170), Point(149,168), Point(151,162), Point(152,160), Point(152,157), Point(152,155), Point(152,151), Point(152,149), Point(152,146), Point(149,142), Point(148,139), Point(145,137), Point(141,135), Point(139,135), Point(134,136), Point(130,140), Point(128,142), Point(126,145), Point(122,150), Point(119,158), Point(117,163), Point(115,170), Point(114,175), Point(117,184), Point(120,190), Point(125,199), Point(129,203), Point(133,208), Point(138,213), Point(145,215), Point(155,218), Point(164,219), Point(166,219), Point(177,219), Point(182,218), Point(192,216), Point(196,213), Point(199,212), Point(201,211)]
]
array_uni_exput = []

   

@window.event
def on_mouse_press(x, y, button, modifiers):
    global points, array_user_input
    points.clear()
    array_user_input.clear()

@window.event
def on_mouse_release(x, y, button, modifiers):
    if len(points) >= 10:
        for i in points:
            array_user_input.append(User_input(i.X, i.Y))
        result = dollar.recognize(points, False)
        print("Result: " + result.Name + " (" + str(round_it(result.Score,2)) + ") in " + str(result.Time) + " ms.")
    else:
        print("Too few points made. Please try again.")

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global points
    if buttons & mouse.LEFT:
        points.append(Point(x, y))

def round_it(n, d):
    d = math.pow(10, d)
    result = (n * d) / d
    return round(result)

@window.event
def on_draw():
    window.clear()
    batch.draw()

for i in array_uni_input:
    for e in i:
        array_uni_exput.append(Unistroke_input(e.X, e.Y))
pyglet.app.run()