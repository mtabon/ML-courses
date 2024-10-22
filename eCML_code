#Euclidian Distance without Loops

def innerproduct(X,Z=None):
    '''
    function innerproduct(X,Z)
    
    Computes the inner-product matrix.
    Syntax:
    D=innerproduct(X,Z)
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    Z: mxd data matrix with m vectors (rows) of dimensionality d
    
    Output:
    Matrix G of size nxm
    G[i,j] is the inner-product between vectors X[i,:] and Z[j,:]
    
    call with only one input:
    innerproduct(X)=innerproduct(X,X)
    '''
    if Z is None: # case when there is only one input (X)
        Z=X;
        
    G = np.dot(X,np.transpose(Z))
        
    return G
    raise NotImplementedError()


def calculate_S(X, n, m):
    '''
    function calculate_S(X)
    
    Computes the S matrix.
    Syntax:
    S=calculate_S(X)
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    n: number of rows in X
    m: output number of columns in S
    
    Output:
    Matrix S of size nxm
    S[i,j] is the inner-product between vectors X[i,:] and X[i,:]
    '''
    assert n == X.shape[0]
    
    
    # YOUR CODE HERE

    Snorm = np.sum(X**2, axis = 1).reshape(-1,1)
    
    S = np.repeat(Snorm, m, axis = 1)
    
    return S

def calculate_R(Z, n, m):
    '''
    function calculate_R(Z)
    
    Computes the R matrix.
    Syntax:
    R=calculate_R(Z)
    Input:
    Z: mxd data matrix with m vectors (rows) of dimensionality d
    n: output number of rows in Z
    m: number of rows in Z
    
    Output:
    Matrix R of size nxm
    R[i,j] is the inner-product between vectors Z[j,:] and Z[j,:]
    '''
    assert m == Z.shape[0]
      
    # YOUR CODE HERE

    Rnorm = np.sum(Z**2, axis=1).reshape(1,-1)
    
    R = np.repeat(Rnorm, n, axis = 0)
    
    return R

def l2distance(X,Z=None):
    '''
    function D=l2distance(X,Z)
    
    Computes the Euclidean distance matrix.
    Syntax:
    D=l2distance(X,Z)
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    Z: mxd data matrix with m vectors (rows) of dimensionality d
    
    Output:
    Matrix D of size nxm
    D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    
    call with only one input:
    l2distance(X)=l2distance(X,X)
    '''
    if Z is None:
        Z=X;

    n,d1=X.shape
    m,d2=Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"

    
    # YOUR CODE HERE
    S = np.sum(X**2, axis = 1)
    R = np.sum(Z**2, axis = 1)
    
    D = np.add.outer(S, R) - 2 * np.dot(X, Z.T)
    
    D[D < 0] = 0.0
    
    D = np.sqrt(D)
    
    return D

    def findknn(xTr,xTe,k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);
    
    Finds the k nearest neighbors of xTe in xTr.
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """
    D = l2distance(xTr, xTe)
    
    indices = np.argsort(D, axis=0)[:k, :]
    dists = np.sort(D, axis=0)[:k, :]
    
    return indices, dists

    def accuracy(truth,preds):
    """
    function output=accuracy(truth,preds)         
    Analyzes the accuracy of a prediction against the ground truth
    
    Input:
    truth = n-dimensional vector of true class labels
    preds = n-dimensional vector of predictions
    
    Output:
    accuracy = scalar (percent of predictions that are correct)
    """
    # YOUR CODE HERE

    truth = np.array(truth).flatten()
    preds = np.array(preds).flatten()
    
    if len(truth)==0 and len(preds) ==0:
        return 0
    
    accuracy = np.mean(truth == preds)
    
    return accuracy

    def knnclassifier(xTr,yTr,xTe,k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);
    
    k-nn classifier 
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    yTr = n-dimensional vector of labels
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    
    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    # fix array shapes
    yTr = yTr.flatten()

    # YOUR CODE HERE
   
    indices, dist = findknn(xTr, xTe, k)
    
    lbls = yTr[indices]
    
    m_result = mode(lbls, axis=0)
    preds = m_result.mode
    
    return preds.flatten()
    
