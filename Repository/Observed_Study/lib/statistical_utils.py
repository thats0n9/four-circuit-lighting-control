

def tcrit(sample_size):
    exact = {8: 2.364624251, 12: 2.20098516, 16: 2.131449546, 24: 2.068657610}
    if sample_size in exact:
        return exact[sample_size]
    degrees = sample_size - 1
    normal = 1.959963984540054
    return normal + (normal**3 + normal)/(4*degrees) + (5*normal**5 + 16*normal**3 + 3*normal)/(96*degrees*degrees) + (3*normal**7 + 19*normal**5 + 17*normal**3 - 15*normal)/(384*degrees**3)
