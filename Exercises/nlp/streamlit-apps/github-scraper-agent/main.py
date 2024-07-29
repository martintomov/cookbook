
#################################### 
##         Import Libraries       ##
####################################

from bs4 import BeautifulSoup
import requests
import pandas as pd

#################################### 
##       Scrape Github Data       ##
####################################

def get_data(user_name):
    # Get the Github user data from user name
    url = "https://github.com/{}?tab=repositories".format(user_name)
    page = requests.get(url)
    if page.status_code == 200:            
        soup = BeautifulSoup(page.content , 'html.parser')
        user_info = {}

        # Getting full name
        user_info['Name'] = soup.find(class_ = 'vcard-fullname').get_text()
        # Getting Image
        user_info['image_url'] = soup.find(class_ = 'avatar')['src']
        # Getting Number of followers and following
        user_info['Followers'] = soup.select_one("a[href*=followers]").get_text().strip().split('\n')[0]
        user_info['Following'] = soup.select_one("a[href*=following]").get_text().strip().split('\n')[0]
        # Getting the location and the url of the personal portfolio or page of contact
        #location
        try:
            user_info['Location'] = soup.select_one('li[itemprop*=home]').get_text().strip()
        except:
            # If there is not location field return empty string
            user_info['Location'] = ''

        #url
        try:
            user_info['URL'] = soup.select_one('li[itemprop*=url]').get_text().strip()
        except:
            # If there is not personal website url field return empty string
            user_info['URL'] = ''
        
        # Getting Repositories
        repositories = soup.find_all(class_ = 'source')                
        repos_info = []

        for repo in repositories:            
            #Name and link of the repository
            try:
                name =  repo.select_one('a[itemprop*=codeRepository]').get_text().strip()                               
                link = 'https://github.com/{}/{}'.format(user_name,name)
            except:
                name = ''
                link = ''
            #Last time updated
            try:
                updated = repo.find('relative-time').get_text()
            except:
                updated = ''
            # programming language used
            try:
                language = repo.select_one('span[itemprop*=programmingLanguage]').get_text()
            except:
                language = ''
            # description of the repository
            try:
                description = repo.select_one('p[itemprop*=description]').get_text().strip()
            except:
                description = ''
            # Add the repository information
            repos_info.append({'name': name ,
                              'link': link ,
                              'updated ':updated ,
                              'language': language ,
                              'description':description
                              })
        # Convert the repository information list to DataFrame    
        repos_info = pd.DataFrame(repos_info)       
        # Return user information and the information of each repository     
        return user_info , repos_info