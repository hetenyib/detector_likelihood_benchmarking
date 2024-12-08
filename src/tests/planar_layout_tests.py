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
from src.layout_planar_main import *
from src.layout_templates import *
from src.layout_direct_borders import add_kesselring_border_plaquettes, add_truncated_links


@pytest.mark.parametrize("index, width, expected", 
                         [(9, 7, (2,1)), 
                          (5, 7, (5, 0)), 
                          (28, 7, (0, 4)), 
                          (11, 9, (2,1)), 
                          (41, 9, (5,4)), 
                          (19, 9, (1, 2))])
def test_coord_from_index(index, width, expected):
    assert coord_from_index(index, width) == expected
# End test_coord_from_index


@pytest.mark.parametrize("width, coord, expected", 
                         [(7, (2,1), 9), 
                          (7, (5, 0), 5), 
                          (7, (0, 4), 28), 
                          (9, (2,1), 11), 
                          (9, (5,4), 41), 
                          (9, (1, 2), 19)])
def test_index_from_coord(width, coord, expected):
    x,y = coord
    assert index_from_coord(x, y, width) == expected
# End test_index_from_coord


@pytest.mark.parametrize('index, width, expected',
                         [(11, 9, [1, 9, 3, 21, 13, 19]),
                          (15, 9, [5,7,17,13,23,25]),
                          (31, 9, [21,29,23,33,41,39]),
                          (9, 7, [1,3,7,11,15,17])])
def test_get_adjacent_link_indexes(index, width, expected):
    assert sorted(get_adjacent_link_indexes(index, width)) == sorted(expected)
# End test_get_adjacent_link_indexes


@pytest.mark.parametrize("index, width, expected", 
                         [(0, 7, -1), 
                          (9, 7, -2), 
                          (15, 7, 1), 
                          (7, 7, 2), 
                          (33, 9, 2), 
                          (41, 9, 0), 
                          (13, 9, 1), 
                          (30, 9, -3), 
                          (24, 9, -1), ])
def test_get_location_type(index, width, expected):
    link = Link(index, width, True, 0)
    assert get_location_type(link.x, link.y) == expected
# End test_get_link_color_type


@pytest.mark.parametrize("index, width, expected", 
                         [(15, 7, 'Y'), 
                          (7, 7, 'Z'), 
                          (1, 7, 'X'), 
                          (33, 9, 'Z'), 
                          (41, 9, 'X'), 
                          (13, 9, 'Z'), 
                          (23, 9, 'Y'), ])
def test_get_link_orientation_type(index, width, expected):
    link = Link(index, width, True, 0)
    assert get_link_orientation_type(link.x, link.y) == expected
# End test_get_link_orientation_type


@pytest.mark.parametrize("index, width, a0, a1", 
                         [(15, 7, 14, 16), 
                          (7, 7, 0, 14), 
                          (1, 7, 0, 2), 
                          (33, 9, 24, 42), 
                          (41, 9, 40, 42), 
                          (13, 9, 4, 22), 
                          (23, 9, 22, 24), ])
def test_get_adj_qubits(index, width, a0, a1):
    link = Link(index, width, True, 0)
    assert link.adj0 == a0 and link.adj1 == a1
# End test_get_adj_qubits


@pytest.mark.parametrize('size',
                         np.arange(2,7))
def test_kesselring_border_plaquettes(size):
    '''
    Tests that every truncated plaquette has two links of each type,
    and that exactly one of those two links is truncated.
    '''
    layout = parallelogram(size, size)

    for tp in layout.truncated_plaquettes:
        truncated_plaq = layout.truncated_plaquettes[tp]
        for color in range(3):
            colored_links = [link 
                             for link in truncated_plaq.all_links
                             if link.color_type == color]
            assert len(colored_links) == 0 or len(colored_links) == 2, 'Wrong number of colored links.'
            if len(colored_links) == 2:
                num_trunc_links = 0
                for link in colored_links:
                    if link.truncated:
                        num_trunc_links += 1
                assert num_trunc_links == 1, 'Wrong number of truncated links.'
# End test_kesselring_border_plaquettes

@pytest.mark.parametrize('width, length, bb_num_t_links, p_num_t_links, bb_num_t_plaqs, p_num_t_plaqs, bb_num_bdr_plaqs, p_num_bdr_plaqs',
                        [(1, 1, 6, 6, 0, 0, 1, 1),
                        (2, 1, 8, 8, 2, 2, 2, 2),
                        (3, 1, 10, 10, 4, 4, 3, 3),
                        (4, 1, 12, 12, 6, 6, 4, 4),
                        (1, 2, 8, 8, 2, 2, 2, 2),
                        (2, 2, 10, 10, 4, 4, 4, 4),
                        (3, 2, 12, 12, 6, 6, 6, 6),
                        (4, 2, 14, 14, 8, 8, 8, 8),
                        (1, 3, 10, 10, 3, 4, 3, 3),
                        (2, 3, 12, 12, 5, 6, 6, 6),
                        (3, 3, 14, 14, 7, 8, 9, 9),
                        (4, 3, 16, 16, 9, 10, 12, 12),
                        (1, 4, 12, 12, 4, 6, 4, 4),
                        (2, 4, 14, 14, 6, 8, 8, 8),
                        (3, 4, 16, 16, 8, 10, 12, 12),
                        (4, 4, 18, 18, 10, 12, 16, 16)])
def test_border_features(width, 
                         length, 
                         bb_num_t_links, 
                         p_num_t_links, 
                         bb_num_t_plaqs, 
                         p_num_t_plaqs, 
                         bb_num_bdr_plaqs, 
                         p_num_bdr_plaqs):
    layout_bb = basic_box(width, length)
    add_truncated_links(layout_bb)
    add_kesselring_border_plaquettes(layout_bb)
    layout_p = parallelogram(width, length)

    # Only care about parallelagram really.
    assert len(layout_p.truncated_links) == p_num_t_links, f'Wrong number ({len(layout_p.truncated_links)}) of truncated links in parallellogram'
    assert len(layout_p.truncated_plaquettes) == p_num_t_plaqs, f'Wrong number ({len(layout_p.truncated_plaquettes)}) of truncated plaquettes in basic_box'
    assert len(layout_bb.truncated_links) == bb_num_t_links, f'Wrong number ({len(layout_bb.truncated_links)}) of truncated links in basic_box'
    assert len(layout_bb.truncated_plaquettes) == bb_num_t_plaqs, f'Wrong number ({len(layout_bb.truncated_plaquettes)}) of truncated plaquettes in basic_box'
    
# End test_border_features
