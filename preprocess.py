import numpy as np

# Data is a n*12 matrix filled with strings
def preprocess(X):
    newX=np.zeros((X.shape[0],50))
    ids=X[:,0]
    PClass=X[:,1]
    names=X[:,2]
    titles=X[:,3]
    sexes=X[:,4]
    ages=X[:,5]
    sibs=X[:,6]
    pachs=X[:,7]
    tickets=X[:,8]
    fares=X[:,9]
    cabins=X[:,10]
    embarkeds=X[:,11]
    index=0
    #ignore ids
    
    #PClass
    newX[:,index]=[int(x) for x in PClass]
    index += 1
    #ignore name
    #title
    ranks=np.unique(([((title.split('.'))[0])[1:] for title in titles]))
    for (k,title) in enumerate(titles) :
        title=((title.split('.'))[0])[1:]
        for (i,rank) in enumerate(ranks) :
            if rank==title: #0 if no, 1 if yes, -1 if missing data
                newX[k,index+i]=1        
            if title=='':
                newX[k,index+i]=-1
    index+=len(ranks)
    #sexes
    for k, sex in enumerate(sexes):
        if sex=='':
            newX[k,index]=-1
        else:
            newX[k,index]=int((sex=="female"))  # 1 for females, 0 for males, -1 if no data
    index += 1
    #ages
    for k, age in enumerate(ages):
        if age=='':
            newX[k,index]=-1
        else:
            newX[k,index]=float(age)            # -1 for no data
    index += 1
    #sibs
    for k, sib in enumerate(sibs):
        if sib=='':
            newX[k,index]=-1
        else:
            newX[k,index]=int(sib)              # -1 for no data
    index += 1
    #pachs
    for k, pach in enumerate(pachs):
        if pach=='':
            newX[k,index]=-1
        else:
            newX[k,index]=int(pach)             # -1 for no data
    index += 1
    #tickets are skipped
    #fares
    for k, fare in enumerate(fares):
        if fare=='':
            newX[k,index]=-1
        else:
            newX[k,index]=float(fare)           # -1 for no data
    index += 1
    #cabins
    keys=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
    for k,cabin in enumerate(cabins):#we write int, then letters
        cabin=cabin.split(' ')[-1] #Only the last cabin matters if we have several
        if (len(cabin)==0):
            newX[k,index]=-1
            for i, key in enumerate(keys):
                newX[k,index+1+i]=-1
        else:
            if len(cabin)==1:
                letter=cabin[0]
                newX[k,index]=-1
            else:
                letter=cabin[0]
                newX[k,index]=int(cabin[1:])
            for (i,key) in enumerate(keys) :
                if key==letter: #0 if no, 1 if yes, -1 if missing data
                    newX[k,index+1+i]=1 #Goes up to 33
    index += len(keys)
    #embarked
    ports=np.unique(embarkeds)
    for k, embarkment in enumerate(embarkeds):
        for (i,port) in enumerate(ports) :
            if port==embarkment: #0 if no, 1 if yes, -1 if missing data
                newX[k,index+i]=1        
            if embarkment=='':
                newX[k,index+i]=-1
    index+=len(ports)
    # Remove rows that are still empty
    newX = np.delete(newX, np.s_[index: (np.shape(newX)[1]) ], 1)
    
    # Normalize all columns, setting missing value to 0
    for i in range((np.shape(newX))[1]):
        column = newX[:,i]
        nonEmpty = column[column != -1]
        mean = np.mean(nonEmpty)
        std = np.std(nonEmpty)
        column[column == -1] = mean
        newX[:,i] = (column - mean)/std
    return newX

'''
At the moment, the shape of the dataset is :
    column 0: classes
    columns 1 to 17: ranks
    columns 18 to 22: sex age sibs pach fare
    column 23: number of cabin
    columns 24 to 31: cabin decks
    columns 32 to 34: ports
Note that the shape might change if the data contains other types of values
Especially, the size will adapt to new or missing titles and new or missing ports
'''