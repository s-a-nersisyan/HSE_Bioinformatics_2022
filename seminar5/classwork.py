# Pairwise alignment
'''
from Bio.Align import PairwiseAligner, substitution_matrices

aligner = PairwiseAligner()
aligner.mode = "global"
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5

seq1 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG"
seq2 = "MVLSGEDKSNIKAAWGKIGGHGAEYGAEALERMFASFPTTKTYFPHFDVSHGSAQVKGHG"
alignments = aligner.align(seq1, seq2)
print(alignments[0])
'''

# Reading fasta sequences
'''
from Bio import SeqIO

sequences = SeqIO.parse("SARS_CoV_2_Russia_unaligned.fasta", "fasta")
for seq in sequences:
    print(seq.seq)
    break
'''

# Reading multiple sequence alignment, protein translation
'''
from Bio import AlignIO

aln = AlignIO.read("SARS_CoV_2_Russia_aligned.fasta", "fasta")

def coord(aln, local):
    seq = aln[0]
    global_ = -1
    for j in range(len(seq)):
        if aln[0][j] != "-":
            global_ += 1

        if global_ == local:
            return j

from SARS_CoV_2_genes import gene_coordinates
start = coord(aln, gene_coordinates["Spike"][0] - 1)
end = coord(aln, gene_coordinates["Spike"][1])

wuhan = aln[0, start:end].seq.replace("-", "")
omicron = aln[6161, start:end].seq.replace("-", "")

from Bio.Seq import Seq
wuhan = Seq(wuhan).translate()
omicron = Seq(omicron).translate()
print(wuhan)
print(omicron)
'''
