# Machine Learning Homework 3: Unsupervised Learning and Dimensionality Reduction

## Dependency

 - Python > 3.5
 - Keras >= 2.1.0
 - sklearn >= 0.19
 - jupyter
 - sqlite3
 
(the code may run with earlier version)

## Running the experiments

To get the same results as the one presented 
on the report you just have to run the python
scripts in the `src` directory.

To run the perceptron with dimension reduction:
```bash
python3 -m src.perceptron
```

To run the perceptron with dimention reduction
and clustering features:
```bash
python3 -m src.clustered_perceptron
```

Both of them will generate `.csv` files in the 
`stats` directory. This to files need to be loaded
into the `plottre/data.db` sqlite database
in order to be used correctly by the different
notebook.

You will find more information on the database
schema on first cell of the different notebooks. 

## Plotting the graphs

The four part of this work are divided on four
different notebooks:

 - Clustering: `plotter/dataset.ipynb`
 - Dimension reduction `plotter/dimesion_reduction.ipynb`
 - Perceptron with dimension reduction `plotter/nn.ipynb`
 - Perceptron with dimension reduction and clustering `plotter/clusper.ipynb`

As said above, the last two need the results
to be computed and loaded into the database.
