# Descriptions

### Plan Recognition as Probabilistic Trace Alignment (preparing to submit)

1. Folder 'stochastic' includes datasets in stochastic setting and experimental codes

    (i) To run Probabilistic Trace Alignment (PTA), implement bellow line in command line

    $ cd ~/PTA4PR/stochastic   # GO TO THE PTA DIRECTORY   
    $ python PTA_Table2.py          # The output file ('table2.csv') indicates the performance of TABLE 2 in our paper   
    $ python PTA.py                 # The output files ('activities_64.csv' & 'grid_navi_64.csv') indiate the performance of TABLE 3 in our paper   
 
 Note that the output files are saved in following directory: "~/output_PTA"


2. Folder 'non_schatistic' includes datasets in non-stochastic setting and experimental codes

    (i) To run Probabilistic Trace Alignment (PTA), implement bellow line in command line (the output indiates the performance of PTA in TABLE 4 of our paper):

    $ cd ~/PTA4PR/non_stochastic   # GO TO THE PTA DIRECTORY   
    $ python PTA.py                     # The output files ('activities_64.csv' & 'grid_navi_64.csv') indiate the performance of TABLE 4 in our paper   
    $ python PTA_BLOCKS_WORLD.py        # The output file ('blocks_world_original.csv') indiates the performance of TABLE 5 in our paper   
