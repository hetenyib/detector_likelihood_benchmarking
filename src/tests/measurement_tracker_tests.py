# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import pytest
from src.measurement_tracker import *


@pytest.mark.parametrize('start, mmts, label',
                         [(5,[1,2,3,4],None),
                          (23,[7,13,54,23],'Green'),
                          (4,[5,35,4,3],'mythical'),
                          (8,[2,67,5,3],'hello'),])
def test_measurement_round(start, mmts, label):
    rd = measurement_round(start, mmts, label)
    assert rd.start_mmt == start
    assert rd.mmts == mmts
    assert rd.label == label
# End test_measurement_round


def test_mtrack_add_measurement():
    mtrack = measurement_tracker()
    a = [1,2,3,4]
    b = [7, 13, 54, 23]
    c = [5,35,4,3]
    d = [2,67,5,3]

    mtrack.add_measurement(a)
    mtrack.add_measurement(b)
    mtrack.add_measurement(c)
    mtrack.add_measurement(d)

    assert mtrack.num_mmts == 16
    assert mtrack.mmt_rounds == 4
    assert mtrack.measurements[2] is not None
    assert mtrack.measurements[1].mmts[2] == 54
    assert mtrack.measurements[3].mmts[-1] == 3
    assert mtrack.measurements[0].mmts[0] == 1
# End test_mtrack_add_measurement



@pytest.mark.parametrize('prev_round',
                         [-1,-2,-3,-4])
def test_mtrack_get_previous_round(prev_round):
    mtrack = measurement_tracker()
    rounds = [[1,2,3,4],
        [7, 13, 54, 23],
        [5,35,4,3],
        [2,67,5,3]]

    mtrack.add_measurement(rounds[0])
    mtrack.add_measurement(rounds[1])
    mtrack.add_measurement(rounds[2])
    mtrack.add_measurement(rounds[3])

    mmt_round = mtrack.get_previous_round(prev_round)
    assert mmt_round.mmts == rounds[prev_round]
    assert mmt_round.start_mmt == 16 + 4 * prev_round
# End test_mtrack_get_previous_round


@pytest.mark.parametrize('fake_links, round, expected',
                         [([1,2,3,4,5], -4, [-16, -15, -14, -13]),
                          [[13], -3, [-11]],
                          [[1,3,5,35,100], -1, [-1, -2]],
                          [[1,3,5,35,100], -2, [-5, -8, -7]],
                          [[1,3,5,35,100], -3, []],
                          [[1,3,5,35,100], -4, [-16, -14]],])
def test_mtrack_get_mmt_targets(fake_links, round, expected):
    mtrack = measurement_tracker()
    a = [1,2,3,4]
    b = [7, 13, 54, 23]
    c = [5,35,4,3]
    d = [2,67,5,3]

    mtrack.add_measurement(a)
    mtrack.add_measurement(b)
    mtrack.add_measurement(c)
    mtrack.add_measurement(d)

    assert mtrack.get_mmt_targets(fake_links, round) == expected
# End test_mtrack_get_mmt_targets
