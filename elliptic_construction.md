# Instructions for processing the Elliptic data set

The Elliptic data set, downloadable from [https://www.kaggle.com/ellipticco/elliptic-data-set](), consists of a dynamic graph that can be used for node classification. It was used, for example, for experimentation by the EvolveGCN paper [https://arxiv.org/abs/1902.10191](). A preprocessed version of the data set can be read by the EvolveGCN code, specifically, the dataloader `elliptic_temporal_dl.py` in [https://github.com/IBM/EvolveGCN](). Here are the preprocessing instructions.


## Step 0: Download data

Download data from [https://www.kaggle.com/ellipticco/elliptic-data-set](). You will see three files `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, and `elliptic_txs_edgelist.csv`.


## Step 1: Create a file named `elliptic_txs_orig2contiguos.csv` and modify `elliptic_txs_features.csv`.

The file `elliptic_txs_features.csv` contains node features, one node each line. Each line contains 167 numbers, where the first number denotes node id. For example, the first three numbers in the first line are

```
230425980,1,-0.1714692896288031
```

Here, `230425980` is the node id. You will replace this id by the line number (starting from 0). Moreover, make the first number and second number floating point numbers. That is, in the modified `elliptic_txs_features.csv`, the three numbers in the first line should be

```
0.0,1.0,-0.1714692896288031
```

In the newly created `elliptic_txs_orig2contiguos.csv`, the first line is the header

```
originalId,contiguosId
```

and the lines that follow contain the id conversion information. For example, the line after the header line should be

```
230425980,0
```


## Step 2: Modify `elliptic_txs_classes.csv`

The file `elliptic_txs_classes.csv` contains node labels. Because we have converted the node ids, we need to modify this file accordingly. We also use numeric values to denote the labels.

Specifically, the classes `unknown`, `1`, and `2` are changed to `-1.0`, `1.0`, and `0`, respectively. For example, the line after the header line is changed from 

```
230425980,unknown
```

to

```
0.0,-1.0
```

Additionally, the header line is never changed. It is always

```
txId,class
```


## Step 3: Create a file named `elliptic_txs_nodetime.csv`

This file will add a time stamp to each node.

The header line is

```
txId,timestep
```

Each line afterward will contain two numbers. The first number is the new node id and the second number is the time stamp.

The time stamp appears in the second column of `elliptic_txs_features.csv`. Recall that the first three numbers in the first line of the original `elliptic_txs_features.csv` are

```
230425980,1,-0.1714692896288031
```

The second number `1` indicates time stamp. We will use a zero based indexing and hence shift this number down by 1.

Therefore, the line after the header line of the new file `elliptic_txs_nodetime.csv` is

```
0,0
```

indicating that the time stamp of node 0 is 0.


## Step 4: Modify `elliptic_txs_edgelist.csv` and rename it to `elliptic_txs_edgelist_timed.csv`

The header line is changed from

```
txId1,txId2
```

to

```
txId1,txId2,timestep
```

For each line that follows, the two numbers indicating old node ids are changed to new node ids, followed by time stamp (as floating point number). These two nodes always have the same time stamp in `elliptic_txs_nodetime.csv` (I recommend doing a sanity check by yourself). Then the edge with these two nodes has a time stamp the same as the node time stamps.

Therefore, the line after the header line in `elliptic_txs_edgelist.csv`:

```
230425980,5530458
```

will be changed to, in `elliptic_txs_edgelist_timed.csv`:

```
0,1,0.0
```

because the new node id for `230425980` is `0` and that for `5530458 ` is `1`. The time stamp for these two nodes are `0`.

With all above preprocessing steps done, you should be able to use the dataloader `elliptic_temporal_dl.py` to load in the data set.
