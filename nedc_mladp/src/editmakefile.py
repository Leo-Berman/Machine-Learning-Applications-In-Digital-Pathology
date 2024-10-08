import os

def main():

    Makefiles = []
    
    for root, dirs, files in os.walk("."):
        print(root,dirs,files)

        if 'Makefile' in files:
            Makefiles.append(root + '/Makefile')

    for x in Makefiles:

        with open(x,'r') as f:
            content = f.read()
            
        content = content.replace('rm -f $(SRC)','')
        content = content.replace('rm -f $(MLADP)/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/bin/$(BFILE) ','')
        content = content.replace('$(MLADP)/Machine-Learning-Applications-In-Digital-Pathology','$(MLADP)')

        with open(x,'w') as f:
            f.write(content)

        

if __name__ == "__main__":
    main()
