@echo off

REM Full command
REM C:\Users\danil\Documents\GitHub\Sentiment-Augmented-Bitcoin-Price-Prediction\demo\scripts\execute_orders.bat
REM C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\demo\scripts\execute_orders.bat

REM Enable virtual environment
call .venv\Scripts\activate

REM Save balance for accounts
REM python -m demo.scripts.execute_orders --account=1 --strategy=1
REM python -m demo.scripts.execute_orders --account=2 --strategy=2 --amount=1000
REM python -m demo.scripts.execute_orders --account=3 --strategy=3 --percentage=0.1
REM python -m demo.scripts.execute_orders --account=4 --strategy=4 --amount=1000
REM python -m demo.scripts.execute_orders --account=5 --strategy=5 --percentage=0.1

C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\.venv\Scripts\python.exe -m demo.scripts.execute_orders --account=1 --strategy=1
C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\.venv\Scripts\python.exe -m demo.scripts.execute_orders --account=2 --strategy=2 --amount=1000
C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\.venv\Scripts\python.exe -m demo.scripts.execute_orders --account=3 --strategy=3 --percentage=0.1
C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\.venv\Scripts\python.exe -m demo.scripts.execute_orders --account=4 --strategy=4 --amount=1000
C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\.venv\Scripts\python.exe -m demo.scripts.execute_orders --account=5 --strategy=5 --percentage=0.1

REM WAIT 
REM timeout /t 100