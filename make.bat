@ECHO OFF
if "%~1"=="" goto BLANK
if "%~1"=="install" goto install
if "%~1"=="clean" goto CLEAN
@ECHO ON

:BLANK
cmake -H. -B_build -A "x64" -DCMAKE_INSTALL_PREFIX="_install" -DWHACK_BUILD_UNITTESTS=ON -DWHACK_BUILD_UNITTESTS_FOR_CUDA=ON
GOTO DONE

:INSTALL
set @location="_install"
if NOT "%2"=="" set @location="%2"
cmake -H. -B_build -A "x64" -DCMAKE_INSTALL_PREFIX=%@location% 
cmake --build _build --config Release --target install
GOTO DONE

:CLEAN
rmdir /Q /S _build
rmdir /Q /S _install
GOTO DONE

:DONE
pause