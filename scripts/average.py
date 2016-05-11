#!/usr/bin/env python
#
# File: average.py
# Purpose: reads an input data file, prints average and st.dev. for input data
#

import math
from ROOT import TCanvas, TH1F, gROOT
gROOT.SetBatch(1)

#def drawText(xpos,ypos,color,text) :
#  txtObj = TLatex(xpos,ypos,text)
#  txtObj.SetTextColor(color)
#  txtObj.SetNDC(1)
#  txtObj.SetTextAlign(11)
#  txtObj.SetTextSize(0.02)
#  txtObj.Draw()
#  return txtObj

#.. parse options
from optparse import OptionParser
usage = "usage: %prog [options] arg1 arg2"
parser = OptionParser(usage=usage)
parser.add_option("-f", "--file", dest="filename", help="input FILE with list of funds to compare", metavar="FILE")
parser.add_option("-v", "--verbose", dest="debug", help="verbose output", action="store_true")
#parser.add_option("-q", "--quiet", dest="debug", help="verbose output", action="store_false", default=False)
#parser.add_option("-y", "--ymin", dest="ymin", help="min value of Y-axis", type="float", default=0)
#parser.add_option("-Y", "--ymax", dest="ymax", help="max value of Y-axis", type="float", default=0)
parser.add_option("-p", "--makeplots", dest="plots", help="make plots", action="store_true", default=False)
(options, args) = parser.parse_args()

if options.debug:
  print "verbose =", options.debug

#.. Read data
if options.debug: print "Reading values from file", options.filename

nval=0
sum=0.
sum2=0.
xhigh=0.
xlow=9999.

datafile = open(options.filename)
#for line in open(options.filename).read().split('\n'):
for line in datafile.read().split('\n'):
  if line=='': continue
  if line=='-.------': continue
  x = float(line)
  sum += x
  sum2 += x*x
  nval += 1
  if x > xhigh:
    xhigh = x
  if x < xlow:
    xlow = x

datafile.close()

#mean = sum/nval
#sigma = math.sqrt(nval*sum2-sum*sum)/(nval-1)
#print "\n Naive average: %f +/- %f (%i points, sigma/mean=%3.1f) xhigh=%f\n" % (mean, sigma, nval, sigma/mean*100, xhigh)

#.. always discard highest data point (outlier)
mean = 0.0
sigma = 0.0
quality = 0.0
if nval>0:
  sum -= xhigh
  sum2 -= xhigh*xhigh
  nval -= 1
  mean = sum/nval
  aux = nval*sum2 - sum*sum
  if aux<0.0:
    aux=0.0
  sigma = math.sqrt(aux)/(nval-1)

if mean>0:
  quality = sigma/mean*100

if options.debug is True:
  print "\n Improved average: %f +/- %f (%i points, sigma/mean=%3.1f)\n" % (mean, sigma, nval, quality)
else:
  print "%f %f %i %3.1f" % (mean, sigma, nval, quality)

#.. fill and save a histogram
if options.plots and xlow<xhigh:
  datafile = open(options.filename)
  hist = TH1F("hist","hist",100,0.9*xlow,1.1*xhigh)
  for line in datafile.read().split('\n'):
    if line=='': continue
    if line=='-.------': continue
    x = float(line)
    hist.Fill(x)

  datafile.close()

  c1 = TCanvas("c1","c1",400,400)
  hist.Draw()
  c1.SaveAs("last.png")
