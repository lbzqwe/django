from django.conf.urls import url

from . import views

app_name = 'polls'
urlpatterns=[
            #CLASS
            url(r'^$',views.IndexView.as_View(),name='inedex'),
            url(r'^(?P<pk>[0-9]+)$',views.DetailView.as_View(),name='detail'),
            url(r'^(?P<pk>[0-9]+)/results/$',views.ResultsView.as_View(),name='results'),
        
            #METHOD
            url(r'^(?P<question_id>[0-9]+)/vote/$',views.vote,name='vote'),
        ]
