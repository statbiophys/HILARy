from glob import glob
import pandas as pd
import os

for fn in glob("data/benchmark/*.csv.gz"):
    df = pd.read_csv(fn)
    ## now create the fasta files for partis
    subsample = df.sample(n=1000)
    with open("data/fasta_benchmark/" + os.path.basename(fn)[:-7] + ".fasta", 'w') as fw:
        for ii, seq in enumerate(subsample.SEQUENCE.to_list()):
            fw.write(">seq" + str(ii) + "\n")
            fw.write(seq + "\n")
    print(fn)
