import tensorflow as tf
import cv2
import io,json
import src

from google.protobuf.json_format import MessageToJson
i = 0
with tf.Session() as sess:
    while i < 1:
        for example in tf.python_io.tf_record_iterator("../datasets/tf_datasets/cad_60_120/cad-60/Person1/train_000.tfrecord"):
            result = tf.train.Example.FromString(example)
            jsonMessage = MessageToJson(tf.train.Example.FromString(example))
            
            with io.open("../datasets/tf_datasets/cad60-small-tmp/train_000.json","w",encoding="utf-8") as f:
                jsonStr=json.dumps(jsonMessage,ensure_ascii=False)
                if isinstance(jsonStr,str):
                    jsonStr=jsonStr.decode("utf-8")
                    f.write(jsonStr)
            f.close()
            
            print ("[**] jsonMessage =")
            print jsonMessage
            print len(jsonMessage)
            i +=1

