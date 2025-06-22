from nectargan.testers.tester import Tester

def main():
    tester = Tester(config=None)
    tester.run_test(image_count=30)

if __name__== "__main__":
    main()