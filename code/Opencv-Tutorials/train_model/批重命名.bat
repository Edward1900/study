@echo off&setlocal EnableDelayedExpansion
color 0a
echo %date% %time%
echo 正在批量重命名文件......
set a=1
for /f "delims=" %%i in ('dir /b *.jpg') do (
    if not "%%~ni" == "%~n0" (
        if !a! LSS 10 (ren "%%i" "!a!.jpg")else (ren "%%i" "!a!.jpg")
        set /a a+=1
    )
)
set /a a-=1
echo 重命名完成，共重命名%a%个文件。
pause
 