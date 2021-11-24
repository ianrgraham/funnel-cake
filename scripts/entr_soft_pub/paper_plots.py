import numpy as np

# plots to be made 
# 1) Prob rearrang softness
# 2) Prob rearrang local excess entropy
# 3) S_{2,i} S_i correlation plot
# 4) S_2 from stratifying by S


def local_g_s2_terms(Xs):
        
    kinda_rdf = Xs[0::2] + Xs[1::2]
    r = np.linspace(0.1,5.0,50)
    r2 = (4*np.pi*np.power(r,2))
    almost_rdf = (kinda_rdf/r2)
    rdf_mean = np.mean(almost_rdf[20:])
    g = almost_rdf/rdf_mean
    s2 = -2*np.pi*np.nan_to_num((g*np.log(g) - g + 1)*r)*(r[1]-r[0])
    return r, g, s2

def softness_terms(Xs, pipe):
    data = [Xs]
    scaled_data = pipe[0].transform(data)

    terms = pipe[1].coef_[0]*scaled_data[0]
    soft_terms = terms[0::2] + terms[1::2]  # combine A and B terms
    return soft_terms