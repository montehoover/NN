file_name = 'log-500-1000r.txt'
id = '500-1000r'

with open(file_name, 'r') as f:
    sq_trn = []
    sq_tst = []
    ls_trn = []
    ls_tst = []
    for line in f:
        l = line.split()

        if 'Squared' in l and 'train:' in l:
            sq_trn.append(l[-1])
        if 'Squared' in l and 'test:' in l:
            sq_tst.append(l[-1])
        if '0/1' in l and 'train:' in l:
            ls_trn.append(l[-1])
        if '0/1' in l and 'test:' in l:
            ls_tst.append(l[-1])

with open('sq_trn{}.txt'.format(id), 'w') as f:
    for n in sq_trn:
        print(n, file=f)

with open('sq_tst{}.txt'.format(id), 'w') as f:
    for n in sq_tst:
        print(n, file=f)

with open('ls_trn{}.txt'.format(id), 'w') as f:
    for n in ls_trn:
        print(n, file=f)

with open('ls_tst{}.txt'.format(id), 'w') as f:
    for n in ls_tst:
        print(n, file=f)
