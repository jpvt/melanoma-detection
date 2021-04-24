sudo docker build -f Dockerfile -t melanoma-api:latest .


sudo docker run -p 12000:12000 -ti melanoma-api:latest python3 api.py