import os, sys, array
import numpy as np

class BigFile:

    def __init__(self, datadir, ndims):
        id_file = os.path.join(datadir, "id.txt")
        self.names = [x.strip() for x in str.split(open(id_file).read()) if x.strip()]
        self.name2index = dict(zip(self.names, range(len(self.names))))
        self.ndims = ndims
        self.binary_file = os.path.join(datadir, "feature.bin")
        print ("[%s] %d instances loaded from %s" % (self.__class__.__name__, len(self.names), datadir))


    def read(self, requested, isname=True):
        requested = set(requested)
        if isname:
            index_name_array = [(self.name2index[x], x) for x in requested if x in self.name2index]
        else:
            assert(min(requested)>=0)
            assert(max(requested)<len(self.names))
            index_name_array = [(x, self.names[x]) for x in requested]
        
        index_name_array.sort(key=lambda v:v[0])
        sorted_index = [x[0] for x in index_name_array]

        nr_of_images = len(index_name_array)
        vecs = [None] * nr_of_images
        offset = np.float32(1).nbytes * self.ndims
        
        res = array.array('f')
        fr = open(self.binary_file, 'rb')
        fr.seek(index_name_array[0][0] * offset)
        res.fromfile(fr, self.ndims)
        
        for next in sorted_index[1:]:
            move = (next-1) * offset
            fr.seek(move, 1)
            res.fromfile(fr, self.ndims)
        
        fr.close()

        return [x[1] for x in index_name_array], [ res[i*self.ndims:(i+1)*self.ndims].tolist() for i in range(nr_of_images) ]


    def read_one(self, name):
        renamed, vectors = self.read([name])
        return vectors[0]    

    def shape(self):
        return [len(self.names), self.ndims]



if __name__ == '__main__':
    bigfile = BigFile('toydata/FeatureData/f1', 3)

    imset = str.split('b z a a b c')
    renamed, vectors = bigfile.read(imset)


    for name,vec in zip(renamed, vectors):
        print name, vec

