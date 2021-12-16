**WZK**
---
![WerkZeugKasten Logo](WerkZeugKasten.png)

Abbreviation of the German noun *"WerkZeugKasten"* (~*"toolbox"*)

---
A mixed collection of "fairly general" convenience functions I use across different project.
The functions are structured with respect to the existing module they extend, or the functionality they facilitate. 
Some examples are:

* numpy
  * wrapper for shapes and axes
* matplotlib (mpl)
  * settings to produce scientific figures (tex + pdf)
  * advanced interactive capabilities
    * draggable patches
    * key listener for sliders
* multiprocessing
  * general wrapper to multiprocess an arbitrary function
* math
  * finite differences
* geometry
* spatial transforms
  * 
* read / write files
* lists, tuples, dicts
* timing


# Attention
This module has some wrapper for multiprocessing and matplotlib
and sets some backend flags, this should not interfere with anything else but be aware of it

* matplotlib
  * set backend
* multiprocessing
  * limit the threads used by numpy / scipy to 1

make sure wzk is the first thing you import especially before matplotlib / numpy

---
This repo has nothing to do with [Werkzeug](https://pypi.org/project/Werkzeug/). 
They just seem to have the same humor.
