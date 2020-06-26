from django.contrib import admin
from django.urls import path,include
admin.site.site_header="Code Hospital DataBase"
admin.site.site_title="ravan database"
admin.site.index_title="Hello Aman Welcome Missing You"


urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include('priceprediction.urls'))
]
