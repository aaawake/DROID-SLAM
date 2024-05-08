#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
The Kinect provides the color and depth images in an un-synchronized way. This means that the set of time stamps from the color images do not intersect with those of the depth images. Therefore, we need some way of associating color images to depth images.

For this purpose, you can use the ''associate.py'' script. It reads the time stamps from the rgb.txt file and the depth.txt file, and joins them by finding the best matches.
"""

import argparse
import sys
import os
import numpy
from tqdm import tqdm

class alignTimestamp():
    def __init__(self, first, second, third, offset=0.0, max_difference=0.05) -> None:
        self.first = first
        self.second = second
        self.third = third
        self.offset = offset
        self.max_difference = max_difference
        
    def read_file_list(self, filename):
        """
        Reads a trajectory from a text file. 
        
        File format:
        The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
        and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
        
        Input:
        filename -- File name
        
        Output:
        dict -- dictionary of (stamp,data) tuples
        
        """
        # print("Start reading file: %s"%(os.path.basename(filename)))
        with open(filename) as file:
            data = file.read()
            lines = data.replace(","," ").replace("\t"," ").split("\n") 
            list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
            list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
        return dict(list)

    def associate(self, first_list, second_list):
        """
        Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
        to find the closest match for every input tuple.
        
        Input:
        first_list -- first dictionary of (stamp,data) tuples
        second_list -- second dictionary of (stamp,data) tuples
        offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference -- search radius for candidate generation

        Output:
        matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
        
        """
        # print("\nStart aligning timestamps")
        first_keys = list(first_list.keys())
        second_keys = second_list.keys()
        potential_matches = [(abs(a - (b + self.offset)), a, b) 
                            #  for i in tqdm(range(len(first_keys)), bar_format='Aligning: {percentage:.1f}%|{bar}|', leave=True) 
                            for a in first_keys
                            for b in second_keys 
                            if abs(a - (b + self.offset)) < self.max_difference]
        # print("Sort")
        potential_matches.sort()
        matches = {}
        first_keys = list(first_keys)
        second_keys = list(second_keys)
        # for i in tqdm(range(len(potential_matches)), bar_format='Get matches: {percentage:.1f}%|{bar}|', leave=True):
            # diff, a, b = potential_matches[i]
        for diff, a, b in potential_matches:
            if a in first_keys and b in second_keys:
                first_keys.remove(a)
                second_keys.remove(b)
                matches[a] = b
        
        return matches

    def del_pic(self, file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()
            image_names = [os.path.basename(line.strip().split()[1]) for line in lines]

        rgb_folder = os.path.join(os.path.dirname(file_name), os.path.basename(file_name)[:-4])
        for filename in os.listdir(rgb_folder):
            if filename not in image_names:
                file_path = os.path.join(rgb_folder, filename)
                os.remove(file_path)

    def align(self):
        print("Aligning timestamp...")
        first_list = self.read_file_list(self.first)
        second_list = self.read_file_list(self.second)
        third_list = self.read_file_list(self.third)

        matches12 = self.associate(second_list, first_list)
        matches23 = self.associate(second_list, third_list)
        t1 = sorted(list(matches12.keys()))
        t2 = sorted(list(matches23.keys()))
        matches = [(matches12[t], t, matches23[t]) for t in t1 if t in t2 ]

        with open(self.first, 'w') as first:
            # for i in tqdm(range(len(matches)), bar_format='Writing aligned data: {percentage:.1f}%|{bar}|', leave=True):
            #     a, b, c = matches[i]
            for a, b, c in matches:
                first.write("%f %s\n"%(a," ".join(first_list[a])))
        with open(self.second, 'w') as second:
            # for i in tqdm(range(len(matches)), bar_format='Writing aligned data: {percentage:.1f}%|{bar}|', leave=True):
            #     a, b, c = matches[i]
            for a, b, c in matches:
                second.write("%f %s\n"%(b," ".join(second_list[b])))
        with open(self.third, 'w') as second:
            # for i in tqdm(range(len(matches)), bar_format='Writing aligned data: {percentage:.1f}%|{bar}|', leave=True):
            #     a, b, c = matches[i]
            for a, b, c in matches:
                second.write("%f %s\n"%(c," ".join(third_list[c])))
        # print("Finish!")
        self.del_pic(self.second)
        self.del_pic(self.third)

