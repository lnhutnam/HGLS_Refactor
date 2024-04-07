import os
import random
import argparse
import shutil

def load_quadruples(inpath):
        """train.txt/valid.txt/test.txt reader
        inpath: File path. train.txt, valid.txt or test.txt of a dataset;
        return:
            quadrupleList: A list
            containing all quadruples([subject/headEntity, relation, object/tailEntity, timestamp]) in the file.
        """
        with open(inpath, "r") as f:
            quadruple_lst = []
            for line in f:
                try:
                    line_split = line.split()
                    head = int(line_split[0])
                    rel = int(line_split[1])
                    tail = int(line_split[2])
                    time = int(line_split[3])
                    ce = int(line_split[3])
                    quadruple_lst.append([head, rel, tail, time, ce])
                except:
                    print(line)
        return quadruple_lst

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Constructing sparse temporal knowledge graphs",
        usage="extract.py [<args>] [-h | --help]",
    )
    parser.add_argument("--data-path", "-d", type=str, default="data/ICEWS14", help="Path to data.")
    parser.add_argument("--save-path", "-s", type=str, default=".", help="Save path for constructed sparse graph.")
    parser.add_argument("--ratio", "-rt", type=float, default=0.2, help="Keeping ratio for graph construction. Default: 0.2. Range: [0.2, 0.3, 0.5]")
    
    args = parser.parse_args()
    print(args)
    train_file = args.data_path + "/train.txt"
    valid_file = args.data_path + "/valid.txt"
    test_file = args.data_path + "/test.txt"
    stat_file = args.data_path + "/stat.txt"
    ent2id_file = args.data_path + "/entity2id.txt"
    rel2id_file = args.data_path + "/relation2id.txt"
    
    train_quadruples = load_quadruples(train_file)
    
    sparse_train_quadruples = random.sample(train_quadruples, int(args.ratio * len(train_quadruples)))
    
    for quad in sparse_train_quadruples[:5]:
        print(quad)
        
    save_path = args.save_path + os.sep + str(args.data_path).split("/")[-1] + "-" + str(100*args.ratio)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    shutil.copy2(valid_file, save_path)
    shutil.copy2(test_file, save_path)
    shutil.copy2(stat_file, save_path)
    shutil.copy2(ent2id_file, save_path)
    shutil.copy2(rel2id_file, save_path)
    
    with open(save_path + "/train.txt", "w") as f:
        for quad in sparse_train_quadruples:
            s = quad[0]
            r = quad[1]
            t = quad[2]
            tau = quad[3]
            ce = quad[4]
            # f.write(f"{s}\t{r}\t{t}\t{tau}\t{-1}\n")
            f.write(f"{s}\t{r}\t{t}\t{tau}\t{ce}\n")
            # f.write(f"{s}\t{r}\t{t}\t{tau}\n")
    f.close()
    
     
    