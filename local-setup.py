import os

if __name__ == "__main__":
    try:
        print(f"Access Key: {os.environ['AWS_ACCESS_KEY_ID']!r}")
        print(f"Secret Key: {os.environ['AWS_SECRET_ACCESS_KEY']!r}")
    except KeyError:
        print("Environment variables not set!")
