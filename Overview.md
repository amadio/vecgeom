Overview
========

What is VecGeom ?
-----------------
VecGeom is a library of that provides geometrical shape primitives, 
ways to describe a model geometry and to navigate within it, 
tailored for use in cutting edge particle transport simulation

Shape primitives have the full set of methods for intersection, 
distance to nearest boundary of volume required for navigation of tracks 
for particle transport simulations.

Associating these shapes with materials, and placing them inside others 
can be used to create a hierarchial description of an arbitrary geometrical setup.

VecGeom is capable of locating the volume to which a point belongs, and of finding the intersection of ray from such a starting point in a chosen direction.  These navigation capabilities are optimised using a new set of acceleration techniques.

A distinguishing feature of VecGeom is that it provides the set of methods 
with SIMD signatures to cope with multiple rays/tracks in one call. 
All appropriate methods of the shape/solid primitives have SIMD versions.
A selected set of navigation methods, for selecting the bounding boxes of 
volumes, also have SIMD versions.

**Not part of document - a list of topics to cover**
List of topics from Google document
- How we create geometry objects via factories, 
- The roles of ‘unplaced’, ‘placed’ and ‘specialised’ classes
- The struct describing each shape
- Navigation techniques/features

Geometry volume primitives 
---------------------------
### Unplaced volumes 
An unplaced volume (class **VUnplacedVolume**) represents a geometry shape 
(primitive) and offers interfaces to query distance, location, containment, etc.
The abstract base class VUnplacedVolume represents an unplaced volume.

The volume is typically placed in its "natural" system of coordinates, e.g. a  the center of coordinates for a sphere and for a 
rectangular parallelilepiped ('box') are at their centers.

This is the same concept as Geant4 'solid' (G4VSolid) and TGeo 'shape' (TGeoShape).

-----------------------------------------
### Logical volumes
A logical volume (class **LogicalVolume**) represents an unplaced volume with 
material and related attributes, and can contain a set of placed volumes inside it (which represent its contents.)



### Placed volumes (solids) and specialisations
A placed volume represents a geometry shape which has been located 
with either at a different position or with a rotation or both.
It is the primary object from which a geometry model of a setup is created. 

For reasons of efficiency different versions exist for each combination, 
depending on whether or not a translation or a rotation exists.

A placed solid corresponds to a Geant4 physical volume (G4VPhysicalVolume) and a TGeo node (TGeoNode).

OLD text **which I believe is wrong** A placed solid does not have a direct correspodance in Geant4 - it combines characteristics of 'solid' (G4VSolid) with the transformation that resides in physical volumes (G4VPhysicalVolume).

### How we create geometry objects via factories
The factory class VolumeFactory is the recommended way to create placed volume with different specialisations.

[ Creating a shape/solid optimally in VecGeom ]

### The struct describing each shape

How to create a geometry setup in VecGeom


### Additional attributes
*The parameters of an unplaced volume are stored in a struct bearing a corresponding name (e.g. HypeStruct for UnplacedHype).  In addition to the parameters typically a number of precomputed expressions of the parameters are also stored, in order to reduce the CPU cost of methods called during tracking*.

Navigation techniques/features
------------------------------



