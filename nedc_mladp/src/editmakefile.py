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
            
        content = content.replace('rm -f $(MLADP)/nedc_mladp/bin/$(BFILE)','')

        with open(x,'w') as f:
            f.write(content)

        

if __name__ == "__main__":
    main()
