# Processing

There are a variety of functions defined by HCIToolbox to process HCI data. In particular, routines like derotating cubes, collapsing data cubes, scaling SDI data, etc. are available.

!!! tip "Multi-threading"
    Many of the methods that work on cubes of data try to multi-thread the operations along the time axis. Make sure you set the environment variable `JULIA_NUM_THREADS` before staring your runtime to take advantage of this.

## Index

```@index
Pages = ["processing.md"]
```

## API/Reference

```@docs
collapse
collapse!
crop
cropview
derotate
derotate!
expand
flatten
shift_frame
shift_frame!
```
