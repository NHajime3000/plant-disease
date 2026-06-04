# plant-disease
server setup

this server is used to test AI agent and generate AI solution for each kind of plant disease(AIGC) 

classification and model comparison function of the apk is unaffectted without server on pc

Step1. download ollama, then open CMD and enter "ollama pull qwen2.5:3b". type "ollama run qwen2.5:3b" to test if the installation is success.

Step2. again in CMD, change the location to the same directory where "server.py" is, then use python to run.

Step3. remember the ip address appeared in the CMD windows, type in the app.

本测试用flask服务器采用ollama + qwen方案

测试用服务器功能主要为使用自设计agent给出对应农作物疾病处理防治解决方案

无测试服务器对apk的病害识别和模型性能比较功能没有影响
