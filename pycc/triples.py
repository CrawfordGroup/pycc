# Various triples formulations; useful for (T) corrections and CC3

def t3_ijk(o, v, i, j, k, t2, F, ERI):
    no = o.stop
    nv = v.stop - no

    t3 = contract('eab,ce->abc', ERI[i,v,v,v], t2[k,j,v,v])
    t3 += contract('eac,be->abc', ERI[i,v,v,v], t2[j,k,v,v])
    t3 += contract('eca,be->abc', ERI[k,v,v,v], t2[j,i,v,v])
    t3 += contract('ecb,ae->abc', ERI[k,v,v,v], t2[i,j,v,v])
    t3 += contract('ebc,ae->abc', ERI[j,v,v,v], t2[i,k,v,v])
    t3 += contract('eba,ce->abc', ERI[j,v,v,v], t2[k,i,v,v])

    t3 -= contract('mc,mab->abc', ERI[j,k,o,v], t2[i,o,v,v])
    t3 -= contract('mb,mac->abc', ERI[k,j,o,v], t2[i,o,v,v])
    t3 -= contract('mb,mca->abc', ERI[i,j,o,v], t2[k,o,v,v])
    t3 -= contract('ma,mcb->abc', ERI[j,i,o,v], t2[k,o,v,v])
    t3 -= contract('ma,mbc->abc', ERI[k,i,o,v], t2[j,o,v,v])
    t3 -= contract('mc,mba->abc', ERI[i,k,o,v], t2[j,o,v,v])

    return t3
