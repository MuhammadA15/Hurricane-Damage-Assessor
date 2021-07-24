import requests 
import gzip
import shutil
import os
from bs4 import BeautifulSoup 
  

  
# specify the URL of the archive here 

  
def get_data_links(archive_url): 
      
    # create response object 
    r = requests.get(archive_url) 
      
    # create beautiful-soup object 
    soup = BeautifulSoup(r.content,'html5lib') 
      
    # find all links on web-page 
    links = soup.findAll('a') 
  

    data_links = [archive_url + link['href'] for link in links if link['href'].endswith('dat')] 
    
  
    return data_links 


def get_archive_data_links(archive_url): 
      
    # create response object 
    r = requests.get(archive_url) 
      
    # create beautiful-soup object 
    soup = BeautifulSoup(r.content,'html5lib') 
      
    # find all links on web-page 
    links = soup.findAll('a') 
  

    data_links = [archive_url + link['href'] for link in links if link['href'].endswith('gz')] 
    
  
    return data_links 

def gz_extract(directory):
    extension = ".gz"
    os.chdir(directory)
    for item in os.listdir(directory): # loop through items in dir
      if item.endswith(extension): # check for ".gz" extension
          gz_name = os.path.abspath(item) # get full path of files
          file_name = (os.path.basename(gz_name)).rsplit('.',1)[0] #get file name for file within
          with gzip.open(gz_name,"rb") as f_in, open(file_name,"wb") as f_out:
              shutil.copyfileobj(f_in, f_out)
          os.remove(gz_name) # delete zipped file
    print("All files Unzipped")   

  
def download_data_series(data_links): 
    global file_name
    for link in data_links: 
  
        '''iterate through all links in file_links
        and download them one by one'''
          
        # obtain filename by splitting url and getting 
        # last string 
        file_name = link.split('/')[-1] 
  
        print( "Downloading file:%s"%file_name) 
          
        # create response object 
        r = requests.get(link, stream = True) 
          
        # download started 
        with open(file_name, 'wb') as f: 
            for chunk in r.iter_content(chunk_size = 1024*1024): 
                if chunk: 
                    f.write(chunk) 
          
    print( "%s downloaded!\n"%file_name )
  
    print ("All files downloaded!")
    
    
    return file_name

if __name__ == "__main__": 
  btkurl = "https://ftp.nhc.noaa.gov/atcf/btk/"
  publicurl = "https://ftp.nhc.noaa.gov/atcf/aid_public/"
  dir_name = 'track_data/atcf/'
   #getting all .dat links for btk url 
  data_links = get_data_links(btkurl) 
  file_name = download_data_series(data_links) 
   #getting all .gz links for public url
  data_links2 = get_archive_data_links(publicurl)
  file_name2 = download_data_series(data_links2)
  gz_extract(dir_name)
     