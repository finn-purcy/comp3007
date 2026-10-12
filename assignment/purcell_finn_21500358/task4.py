

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Finn J. Purcell
# Last Modified: 10/10/2026

import os
import cv2
import numpy as np
from task1 import *
from task2 import *
from task3 import *
from utilities import *
import shutil

def read_from_images(object_dir):
    txtPaths = find_all_files(object_dir)
    
    if len(txtPaths) > 2: #lcd branch
        num_digits = len(txtPaths)
        digits = np.empty(num_digits, dtype=int)

        for txtPath in txtPaths:
            txtPath = txtPath[0]
            extension_index = txtPath.find('.')
            digit_number = int(txtPath[extension_index-1])

            with open(txtPath) as txt:
                digit_value = int(txt.readline())

            digits[digit_number-1] = digit_value

        #some ugly formatting code
        systolic = int(''.join(map(str, list(digits[:3]))))
        diastolic = int(''.join(map(str, list(digits[3:5]))))
        hrate = int(''.join(map(str, list(digits[5:]))))

        return f"bpm {systolic},{diastolic},{hrate}"
    
    else:
        return "thermo 12"




def run_task4(image_path, config):
    home_dir = os.getcwd()

    config_data = read_config(config, 4)
    bug_fix_mode = config_data['bug_fixing']

    os.mkdir("t4temp")


    individual_image_paths = find_all_files(image_path)
    temp_image_paths = []
    for image in individual_image_paths:

        os.makedirs(f"t4temp/{image[1][:image[1].find('.')]}")
        shutil.copyfile(image[0], f"t4temp/{image[1][:image[1].find('.')]}/{image[1]}")

        temp_image_paths.append(f"t4temp/{image[1][:image[1].find('.')]}")
            

    for image in temp_image_paths:
        negative_image_flag = False
        
        run_task1(image, config, task4mode=True)
        os.chdir(image)


        try:
            run_task2("task1", config, task4mode=True)
            run_task3("task2", config, task4mode=True)
            reading_string = read_from_images("task3")

        except FileNotFoundError:
            print("Detected Negative Image")
            negative_image_flag = True
            if not bug_fix_mode:
                import traceback
                traceback.print_exc()

        os.chdir(home_dir)

        if not negative_image_flag:
            save_output(f"{config_data['output_path']}/{image[image.find('/'):]}.txt", reading_string)

    
    if not bug_fix_mode:
        shutil.rmtree("t4temp")

if __name__ == "__main__":
    run_task4(sys.argv[1], sys.argv[2])