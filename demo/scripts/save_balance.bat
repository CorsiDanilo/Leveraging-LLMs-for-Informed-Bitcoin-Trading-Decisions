@echo off

REM Full command
REM C:\Users\danil\Documents\GitHub\Sentiment-Augmented-Bitcoin-Price-Prediction\demo\scripts\save_balance.bat
REM C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\demo\scripts\save_balance.bat

REM Enable virtual environment
call .venv\Scripts\activate

REM Save balance for accounts
REM python -m demo.scripts.save_balance --account=1
REM python -m demo.scripts.save_balance --account=2
REM python -m demo.scripts.save_balance --account=3
REM python -m demo.scripts.save_balance --account=4
REM python -m demo.scripts.save_balance --account=5

C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\.venv\Scripts\python.exe -m demo.scripts.save_balance --account=1
C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\.venv\Scripts\python.exe -m demo.scripts.save_balance --account=2
C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\.venv\Scripts\python.exe -m demo.scripts.save_balance --account=3
C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\.venv\Scripts\python.exe -m demo.scripts.save_balance --account=4
C:\Users\User1\Desktop\Sentiment-Augmented-Bitcoin-Price-Prediction\.venv\Scripts\python.exe -m demo.scripts.save_balance --account=5

REM WAIT FOR 5 SECONDS
REM timeout /t 5