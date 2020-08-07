Read 3d geometry files stl or obj
Create facet list

// could parallelize this with threads
Loop through facet list
    Loop over voxel space for facet i
    For each voxel, compute intersection with facet
        If intersected
            Store voxel coordinate

For all voxel coordinates
    Write out to stl or obj files

