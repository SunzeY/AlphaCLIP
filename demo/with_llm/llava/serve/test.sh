proxy_on 

python -m llava/serve/gradio_web_server.py --controller http://localhost:10000 --model-list-mode reload &


proxy_off

python -m llava/serve/model_worker.py --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b &

proxy_off
python -m llava.serve.controller --host 0.0.0.0 --port 10000