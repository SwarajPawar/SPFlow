REFERENCE SOURCE
================

IDES: influence diagram based expert system
Alice M. Agogino, Ashutosh Rege 
Mathematical Modelling, Volume 8, 1987, Pages 227-233


DESCRIPTION OF DATA
===================

The Diagnostician’s Problem is to decide which component should be tested for failure first and repaired if appropriate. In an automated
assembly operation, this could mean deciding where the
computer should be sent for “rework”. If the diagnostician
is wrong in his or her decision for rework, valuable time
would have been wasted by the rework technician and hence
in producing the final product. For purposes of illustration,
let us assume that the costs of rework, which involves opening
the computer and pulling out the appropriate board,
testing, and repairing it, is a constant for each board. The
inguence diagram in Figure 8 has been modified to show
these decision and value nodes. Recall that there is an
implicit arc between all decision nodes and the value node.
This “No-Forgetting Arc” has been added to the influence
diagram in Figure 8. It has been assumed that a system
failure has occurred and this information is known at the
time that the rework decision is made.
The diagnostician has two choices given that the system
status shows a failure (E =S =.S,): (1) DL = Send the logic
board to rework first and (2) DI/O = Send the l/O board to
rework first

Each of the two decisions has three possible outcomes
corresponding to the possible states of the boards given that
a system failure has occurred S = So: (1) L0 I/Oo, (2) L1 I/O0,
and (3) L0 I/O1. If the wrong board is sent to rework, then
debug time has been wasted on that board and the other
board must be sent to be debugged and reworked. It is also
possible that the board sent to rework has failed but the
other board has failed also and must be repaired. 

Furthermore, the outcome of the rework decision
also influences the cost. This is due to the fact that
though we pick one of the two boards (say I/O) for rework,
it could well turn out that the other board (L in this case)
has really failed. In such a case, the cost of debugging and
reworking the second board would have to be added to the
original expenditure. Given that a system failure has
occurred. which board should be tested first?



VARIABLES
=========




PARTIAL ORDER
=============