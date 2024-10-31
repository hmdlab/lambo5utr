
class ConstructBase():
    def __init__(self):
        self.seq_parts = {}

    def get_seq(self):
        upstream_seq = ''
        downstream_seq = ''

        flg = 0
        for k, v in self.seq_parts.items():
            if k == 'random_5utr':
                flg = 1
                continue
            if flg == 0:
                upstream_seq += v
            else:
                downstream_seq += v
        return upstream_seq, downstream_seq

class ConstructSample2019EGFP(ConstructBase):
    def __init__(self):
        super().__init__()
        self.seq_parts.update({
            'primer_fwd':'GGGACATCGTAGAGAGTCGTACTTA',
            'random_5utr':'N'*50,
            'egfp_cds':'atgggcgaattaagtaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccaagctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaag',
            'egfp_3end':'ttcgaataaagctag',
            'bgh_signal':'cgcctcgactgtgccttctagttgccagccatctgttgtttg',
            'polyA':'A'*70
        })
        self.name = 'ConstructSample2019EGFP'

class ConstructSample2019mCherry(ConstructBase):
    def __init__(self):
        super().__init__()
        self.seq_parts.update({
            'primer_fwd':'GGGACATCGTAGAGAGTCGTACTTA',
            'random_5utr':'N'*50,
            'mcherry_cds':'atgcctcccgagaagaagatcaagagcgtgagcaagggcgaggaggataacatggccatcatcaaggagttcatgcgcttcaaggtgcacatggagggctccgtgaacggccacgagttcgagatcgagggcgagggcgagggccgcccctacgagggcacccagaccgccaagctgaaggtgaccaagggtggccccctgcccttcgcctgggacatcctgtcccctcagttcatgtacggctccaaggcctacgtgaagcaccccgccgacatccccgactacttgaagctgtccttccccgagggcttcaagtgggagcgcgtgatgaacttcgaggacggcggcgtggtgaccgtgacccaggactcctccctgcaggacggcgagttcatctacaaggtgaagctgcgcggcaccaacttcccctccgacggccccgtaatgcagaagaagaccatgggctgggaggcctcctccgagcggatgtaccccgaggacggcgccctgaagggcgagatcaagcagaggctgaagctgaaggacggcggccactacgacgctgaggtcaagaccacctacaaggccaagaagcccgtgcagctgcccggcgcctacaacgtcaacatcaagttggacatcacctcccacaacgaggactacaccatcgtggaacagtacgaacgcgccgagggccgccactccaccggcggcatggacgagctgtacaag',
            'mcherry_3end':'tcttaa',
            'bgh_signal':'cgcctcgactgtgccttctagttgccagccatctgttgtttg',
            'polyA':'A'*70
        })
        self.name = 'ConstructSample2019mCherry'
    
    