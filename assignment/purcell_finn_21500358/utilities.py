import os
import cv2
import configparser


def save_output(output_path, content, output_type='txt'):

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")

    elif output_type == 'image':
        success = cv2.imwrite(output_path, content)
        if success:
            print(f"Image saved at: {output_path}")
        else:
            print(f"Failed to save image: {output_path}")

    else:
        print("Unsupported output type. Use 'txt' or 'image'.")


def find_all_files(image_path):
    imPaths = []
    for entry in os.listdir(image_path):
        full_path = os.path.join(image_path, entry)

        if os.path.isfile(full_path):
            imPaths.append((full_path, entry))
        elif os.path.isdir(full_path):
            imPaths += (find_all_files(full_path))

    return imPaths

def read_config(config_path, tasknumber):
    config = configparser.ConfigParser()

    config.read(config_path)

    match tasknumber:
        case 1:
            model_path = config.get('Task 1', 'model_path')
            output_path = config.get('Task 1', 'output_path')

            config_values = {'model_path': model_path, 'output_path': output_path}

        case 2:
            output_path = config.get('Task 2', 'output_path')

            config_values = {'output_path': output_path}

        case 3:
            model_path = config.get('Task 3', 'model_path')
            output_path = config.get('Task 3', 'output_path')

            config_values = {'model_path': model_path, 'output_path': output_path}

        case 4:
            bug_fixing = config.getboolean('Task 4', 'bug_fixing')
            output_path = config.get('Task 4', 'output_path')

            config_values = {'bug_fixing': bug_fixing, 'output_path': output_path}

    return config_values