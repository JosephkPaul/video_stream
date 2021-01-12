import os
from os.path import dirname, join
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from streamapp.camera import VideoCamera
from django.template.loader import render_to_string
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from django.template import loader, Context
#path='/staticfiles'
cloud_config= {
        'secure_connect_bundle': join(dirname(__file__), "secure-connect-database.zip")
}
auth_provider = PlainTextAuthProvider('Datauser', 'database@1')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

row = session.execute("select release_version from system.local").one()
if row:
    print(row[0])
else:
    print("An error occurred.")
session.set_keyspace('data')

############

ids=[]


def index(request):
    return render(request, 'streamapp/input.html')
def quiz(request):

    return render(request, 'streamapp/copyquiz.html',{'a':3,'q':{"QUESTION 1 THROUGH REQ":['1','2','3','4'],"QUESTION 2 THROUGH REQ":['12','22','32','42']}})
def take(request):
    global ids

    user = request.POST['user']
    b = request.POST['pass']
    ids.append(user)
    futures = []
    print(user,"take")
    query = "SELECT id FROM userdata"
    futures.append(session.execute_async(query))
    l = []
    for future in futures:
        rows = future.result()
        l.append(str(rows[0].id))
    if user in l:
        c = "SOORY" + " " + "PLEASE LOGIN"
        return render(request, 'streamapp/wrong.html', {'res': c})
    else:
        insert_statement = session.prepare("INSERT INTO userdata (id,pass) VALUES (?,?)")
        session.execute(insert_statement, [user, b])
        c = "WELCOME" + " " + user
        return render(request, 'streamapp/home.html', {'res': c})

def add(request):
    global dpass
    global ids
    global user
    dpass=""
    user=request.POST['user']
    passl=request.POST['pass']
    if False:
        print('cart id exists')
    else:
        query = "SELECT * FROM userdata WHERE id=%s"
        a=session.execute_async(query, [user])
        #for future in futures:
        rows = a.result()
        print(rows[0].field_2_)
        dpass=rows[0].field_2_
        if passl==dpass:
            c="WELCOME"+" "+user
            return render(request,'streamapp/home.html',{'res':c})
        else:
            c = "WELCOME" + " " + user+"WRONG PASSWORD PLEASE TRY AGAIN"
            return render(request, 'streamapp/wrong.html', {'res': c})
Test=False
frame1=""
def gen(camera):
    global frame1
    global Test
    global user
    while True:
        if Test==True:
            print(frame1)
            session.set_keyspace('data')
            insert_statement = session.prepare("INSERT INTO userdata (id,marks) VALUES (?,?)")
            session.execute(insert_statement, [user,str(frame1)])
            print("finaly break")
            break
        frame,frame1 = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def video_fee(request):
        global Test
        print("I AM THERE")
        Test=True
        c=request.POST.get('num1')
       # c = "WELCOME" + " " +"THIS IS YOUR TEST SUMMARY"
        futures = []
        query = "SELECT * FROM userdata WHERE id=%s"
        futures.append(session.execute_async(query,[user]))
        l = []
        for future in futures:
            rows = future.result()
            l.append(str(rows[0].marks))
            a=str(rows[0].marks).split(" ")
            print(l,a)
            c=a

        return render(request, 'streamapp/marks.html', {'c': a})

def video_feed(request):
    print(Test)
    if not(Test):
        return StreamingHttpResponse(gen(VideoCamera()),
                        content_type='multipart/x-mixed-replace; boundary=frame')
    print("video freed")
    return render(request, 'streamapp/wrong.html', {'res': "bryeu", 'data': False})

