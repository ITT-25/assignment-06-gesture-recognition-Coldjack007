# $1 gesture recognizer 

import math
import time

class Point:
    def __init__(self, x, y):
        self.X = x
        self.Y = y

#Constants and Variables
NumUnistrokes = 5
NumPoints = 64
SquareSize = 250.0
Origin = Point(0,0)
Diagonal = math.sqrt(SquareSize * SquareSize + SquareSize * SquareSize)
HalfDiagonal = 0.5 * Diagonal
AngleRange = math.radians(45.0)
AnglePrecision = math.radians(2.0)
Phi = 0.5 * (-1.0 + math.sqrt(5.0)) #Golden Ratio

#####################################################################
#Classes

class Rectangle:
    def __init__(self, x, y, width, height):
        self.X = x
        self.Y = y
        self.Width = width
        self.Height = height

class Unistroke_with_Translate:
    def __init__(self, name, points):
        self.Name = name
        SCREEN_HEIGHT = 400
        for e in range(len(points)):
            percentage_y = points[e].Y / SCREEN_HEIGHT
            flipped_y = (1.0 - percentage_y) * SCREEN_HEIGHT
            flipped_point = Point(points[e].X, flipped_y)
            points[e] = flipped_point
        self.Points_resampled = resample(points, NumPoints)
        radians = indicativeAngle(self.Points_resampled)
        self.Points_rotated = rotateBy(self.Points_resampled, -radians)
        self.Points_scaled = scaleTo(self.Points_rotated, SquareSize)
        self.Points_translated = translateTo(self.Points_scaled, Origin)
        self.Vector = vectorize(self.Points_translated)  #for Protractor

class Unistroke:
    def __init__(self, name, points):
        self.Name = name
        self.Points_resampled = resample(points, NumPoints)
        radians = indicativeAngle(self.Points_resampled)
        self.Points_rotated = rotateBy(self.Points_resampled, -radians)
        self.Points_scaled = scaleTo(self.Points_rotated, SquareSize)
        self.Points_translated = translateTo(self.Points_scaled, Origin)
        self.Vector = vectorize(self.Points_translated)  #for Protractor

class Result:
    def __init__(self, name, score, ms):
        self.Name = name
        self.Score = score
        self.Time = ms

class DollarRecognizer:
    def __init__(self):
        self.Unistrokes = [None] * NumUnistrokes
        self.Unistrokes[0] = Unistroke_with_Translate("rectangle", [Point(78,149),Point(78,153), Point(78,157), Point(78,160), Point(79,162), Point(79,164), Point(79,167), Point(79,169), Point(79,173), Point(79,178), Point(79,183), Point(80,189), Point(80,193), Point(80,198), Point(80,202), Point(81,208), Point(81,210), Point(81,216), Point(82,222), Point(82,224), Point(82,227), Point(83,229), Point(83,231), Point(85,230), Point(88,232), Point(90,233), Point(92,232), Point(94,233), Point(99,232), Point(102,233), Point(106,233), Point(109,234), Point(117,235), Point(123,236), Point(126,236), Point(135,237), Point(142,238), Point(145,238), Point(152,238), Point(154,239), Point(165,238), Point(174,237), Point(179,236), Point(186,235), Point(191,235), Point(195,233), Point(197,233), Point(200,233), Point(201,235), Point(201,233), Point(199,231), Point(198,226), Point(198,220), Point(196,207), Point(195,195), Point(195,181), Point(195,173), Point(195,163), Point(194,155), Point(192,145), Point(192,143), Point(192,138), Point(191,135), Point(191,133), Point(191,130), Point(190,128), Point(188,129), Point(186,129), Point(181,132), Point(173,131), Point(162,131), Point(151,132), Point(149,132), Point(138,132), Point(136,132), Point(122,131), Point(120,131), Point(109,130), Point(107,130), Point(90,132), Point(81,133), Point(76,133)])
        self.Unistrokes[1] = Unistroke_with_Translate("circle", [Point(127,141),Point(124,140), Point(120,139), Point(118,139), Point(116,139), Point(111,140), Point(109,141), Point(104,144), Point(100,147), Point(96,152), Point(93,157), Point(90,163), Point(87,169), Point(85,175), Point(83,181), Point(82,190), Point(82,195), Point(83,200), Point(84,205), Point(88,213), Point(91,216), Point(96,219), Point(103,222), Point(108,224), Point(111,224), Point(120,224), Point(133,223), Point(142,222), Point(152,218), Point(160,214), Point(167,210), Point(173,204), Point(178,198), Point(179,196), Point(182,188), Point(182,177), Point(178,167), Point(170,150), Point(163,138), Point(152,130), Point(143,129), Point(140,131), Point(129,136), Point(126,139)])
        self.Unistrokes[2] = Unistroke_with_Translate("check", [Point(91,185),Point(93,185), Point(95,185), Point(97,185), Point(100,188), Point(102,189), Point(104,190), Point(106,193), Point(108,195), Point(110,198), Point(112,201), Point(114,204), Point(115,207), Point(117,210), Point(118,212), Point(120,214), Point(121,217), Point(122,219), Point(123,222), Point(124,224), Point(126,226), Point(127,229), Point(129,231), Point(130,233), Point(129,231), Point(129,228), Point(129,226), Point(129,224), Point(129,221), Point(129,218), Point(129,212), Point(129,208), Point(130,198), Point(132,189), Point(134,182), Point(137,173), Point(143,164), Point(147,157), Point(151,151), Point(155,144), Point(161,137), Point(165,131), Point(171,122), Point(174,118), Point(176,114), Point(177,112), Point(177,114), Point(175,116), Point(173,118)])
        self.Unistrokes[3] = Unistroke_with_Translate("delete", [Point(123,129),Point(123,131), Point(124,133), Point(125,136), Point(127,140), Point(129,142), Point(133,148), Point(137,154), Point(143,158), Point(145,161), Point(148,164), Point(153,170), Point(158,176), Point(160,178), Point(164,183), Point(168,188), Point(171,191), Point(175,196), Point(178,200), Point(180,202), Point(181,205), Point(184,208), Point(186,210), Point(187,213), Point(188,215), Point(186,212), Point(183,211), Point(177,208), Point(169,206), Point(162,205), Point(154,207), Point(145,209), Point(137,210), Point(129,214), Point(122,217), Point(118,218), Point(111,221), Point(109,222), Point(110,219), Point(112,217), Point(118,209), Point(120,207), Point(128,196), Point(135,187), Point(138,183), Point(148,167), Point(157,153), Point(163,145), Point(165,142), Point(172,133), Point(177,127), Point(179,127), Point(180,125)])
        self.Unistrokes[4] = Unistroke_with_Translate("pigtail", [Point(81,219),Point(84,218), Point(86,220), Point(88,220), Point(90,220), Point(92,219), Point(95,220), Point(97,219), Point(99,220), Point(102,218), Point(105,217), Point(107,216), Point(110,216), Point(113,214), Point(116,212), Point(118,210), Point(121,208), Point(124,205), Point(126,202), Point(129,199), Point(132,196), Point(136,191), Point(139,187), Point(142,182), Point(144,179), Point(146,174), Point(148,170), Point(149,168), Point(151,162), Point(152,160), Point(152,157), Point(152,155), Point(152,151), Point(152,149), Point(152,146), Point(149,142), Point(148,139), Point(145,137), Point(141,135), Point(139,135), Point(134,136), Point(130,140), Point(128,142), Point(126,145), Point(122,150), Point(119,158), Point(117,163), Point(115,170), Point(114,175), Point(117,184), Point(120,190), Point(125,199), Point(129,203), Point(133,208), Point(138,213), Point(145,215), Point(155,218), Point(164,219), Point(166,219), Point(177,219), Point(182,218), Point(192,216), Point(196,213), Point(199,212), Point(201,211)])

    def recognize(self, points, useProtractor):
        t0 = time.time()
        candidate = Unistroke("", points)

        u = -1
        b = float('inf')

        for i, tmpl in enumerate(self.Unistrokes):
            if useProtractor:
                d = optimalCosineDistance(tmpl.Points_translated, candidate.Points_translated)
            else:
                d = distanceAtBestAngle(candidate.Points_translated, tmpl, -AngleRange, +AngleRange, AnglePrecision)
            if d < b:
                b = d
                u = i

        t1 = time.time()
        if u == -1:
            return Result("No match.", 0.0, (t1 - t0) * 1000)
        else:
            score = (1.0 - b) if useProtractor else (1.0 - b / HalfDiagonal)
            return Result(self.Unistrokes[u].Name, score, (t1 - t0) * 1000)
        
def optimalCosineDistance(v1, v2): #for Protractor
    a = 0.0
    b = 0.0
    for i in range(0, len(v1), 2):
        a += v1[i] * v2[i] + v1[i+1] * v2[i+1]
        b += v1[i] * v2[i+1] - v1[i+1] * v2[i]
    angle = math.atan2(b, a)
    return math.acos(a * math.cos(angle) + b * math.sin(angle))

def distanceAtBestAngle(points, T, a, b, threshold):
    x1 = Phi * a + (1.0 - Phi) * b
    f1 = distanceAtAngle(points, T, x1)
    x2 = (1.0 - Phi) * a + Phi * b
    f2 = distanceAtAngle(points, T, x2)
    while (abs(b - a) > threshold):
        if (f1 < f2):
            b = x2
            x2 = x1
            f2 = f1
            x1 = Phi * a + (1.0 - Phi) * b
            f1 = distanceAtAngle(points, T, x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - Phi) * a + Phi * b
            f2 = distanceAtAngle(points, T, x2)
    return min(f1, f2)

def distanceAtAngle(points, T, radians):
    newpoints = rotateBy(points, radians)
    return pathDistance(newpoints, T.Points_translated)

def rotateBy(points, radians):
    c = centroid(points)
    cos_r = math.cos(radians)
    sin_r = math.sin(radians)
    new_points = []
    for p in points:
        qx = (p.X - c.X) * cos_r - (p.Y - c.Y) * sin_r + c.X
        qy = (p.X - c.X) * sin_r + (p.Y - c.Y) * cos_r + c.Y
        new_points.append(Point(qx, qy))
    return new_points

def centroid(points):
    x = sum(p.X for p in points) / len(points)
    y = sum(p.Y for p in points) / len(points)
    return Point(x, y)

def boundingBox(points):
    minX = min(p.X for p in points)
    minY = min(p.Y for p in points)
    maxX = max(p.X for p in points)
    maxY = max(p.Y for p in points)
    return Rectangle(minX, minY, maxX - minX, maxY - minY)

def pathDistance(pts1, pts2):
    return sum(distance(pts1[i], pts2[i]) for i in range(len(pts1))) / len(pts1)

def distance(p1, p2):
        dx = p2.X - p1.X
        dy = p2.Y - p1.Y
        return math.sqrt(dx * dx + dy * dy)

def resample(points, n):
    I = pathLength(points) / (n-1) #interval length
    D = 0.0
    newpoints = [points[0]]

    i = 1
    while i < len(points):
        d = distance(points[i - 1], points[i])
        if (D + d) >= I:
            t = (I - D) / d
            qx = points[i - 1].X + t * (points[i].X - points[i - 1].X)
            qy = points[i - 1].Y + t * (points[i].Y - points[i - 1].Y)
            q = Point(qx, qy)
            newpoints.append(q)
            points.insert(i, q)
            D = 0.0
        else:
            D += d
        i += 1
    if len(newpoints) == n - 1:
        newpoints.append(points[-1])
    return newpoints

def indicativeAngle(points):
    c = centroid(points)
    return math.atan2(c.Y - points[0].Y, c.X - points[0].X)

def scaleTo(points, size): #non-uniform scale; assumes 2D gestures (i.e., no lines)
    B = boundingBox(points)
    new_points = []
    for p in points:
        qx = p.X * (size / B.Width)
        qy = p.Y * (size / B.Height)
        new_points.append(Point(qx, qy))
    return new_points

def translateTo(points, pt): #translates points' centroid
    c = centroid(points)
    new_points = []
    for p in points:
        qx = p.X + pt.X - c.X
        qy = p.Y + pt.Y - c.Y
        new_points.append(Point(qx, qy))
    return new_points

def vectorize(points): #for Protractor
    vector = []
    sum_sq = 0.0
    for p in points:
        vector.extend([p.X, p.Y])
        sum_sq += p.X * p.X + p.Y * p.Y
    magnitude = math.sqrt(sum_sq)
    return [v / magnitude for v in vector]

def pathLength(points):
    d = 0.0
    for i in range(1, len(points)):
        d += distance(points[i - 1], points[i])
    return d