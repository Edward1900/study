import base64
import struct
import os


def read_line(line):
    m_id, image_search_rank, image_url, page_url, face_id, face_rectangle, face_data = line.split("\t")
    rect = struct.unpack("ffff", base64.b64decode(face_rectangle))
    return m_id, image_search_rank, image_url, page_url, face_id, rect, base64.b64decode(face_data)


def write_image(filename, data):
    with open(filename, "wb") as f:
        f.write(data)


def unpack(file_name, output_dir):
    i = 0
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            m_id, image_search_rank, image_url, page_url, face_id, face_rectangle, face_data = read_line(line)
            img_dir = os.path.join(output_dir, m_id)
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            img_name = "%s-%s" % (image_search_rank, face_id) + ".jpg"
            write_image(os.path.join(img_dir, img_name), face_data)
            i += 1
            if i % 1000 == 0:
                print(i, "images finished")
        print("all finished")


def main():
    file_name = "F:\\FaceImageCroppedWithAlignment.tsv"
    output_dir = "H:\\MS_Celeb_1M\\"
    unpack(file_name, output_dir)


if __name__ == '__main__':
    main()