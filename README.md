# Projekt zaliczeniowy z wdsi2018

Celem projektu jest stworzenie programu uczącego się na podstawie danych wejściowych. Danymi wejściowymi są parametry komputerów i ich zmierzona eksperymentalnie wydajność.

### Konsultacje z dnia 21.04.2018

- Dane mają zostać jeszcze bardziej uszczuplone - ostatnia kolumna atrybutów jest niepotrzebna, natomiast przedostatniej nie normalizujemy.
- W projekcie mamy zastosować bibliotekę libSVM.
- Dane do krzyżowania mają zostać pomieszane (kolejność rekordów), i podzielone na 5 zbiorów,
- Uczymy jednym zbiorem, a następnie testujemy pozostałymi zbiorami. Robimy tak dla każdego zbioru.

### Konfiguracja środowiska

Program wymaga do swojego działania biblioteki libsvm, którą należy sklonować do katalogu z projektem. https://github.com/cjlin1/libsvm.git

Bibliotekę należy skompilować (make) i to samo należy zrobić dla interfejsu python'a (podkatalog python). 
Następnie należy dodać linki symboliczne do:
* svm-train
* svm-predict
* svm-scale
* libsvm.so.2

Dowiązania należy utworzyć w którymś z katalogów wchodzących w skład systemowego $PATH'a, np /usr/bin. Po wykonaniu tych kroków, powinniśmy być w stanie wywoływać programy biblioteki libsvm, bez konieczności odnoszenia się do ich ścieżki bezwzględnej. Dzięki temu działać zacznie również interfejs python'a, a tym samym program app.py.

### Korzystanie z programu

Program został napisany tak, aby za jego pomocą dało się najpierw pomieszać dane. Poniższe polecenie korzysta z pliku data.dat jako źródła danych, a dane pomieszane w takiej samej formie (CSV) zapisuje do pliku którego nazwę podajemy jako wartość opcji '-w':

```
python app.py -w shuffled.data
```
Jeżeli natomiast chcemy przekazać nasz pomieszany zestaw danych do programu, wówczas musimy wywołać program z opcją '-i', której wartością jest nazwa pliku, który chcemy otworzyć:

```
python app.py -i shuffled.data
```

Aby wywołać program z domyślnym zestawem danych (nie pomieszanym!), wystarczy po prostu uruchomić go bez parametrów:

```
python app.py
```
Program realizuje funkcję regresji. Zgodnie z założeniami projektowymi typ jądra to RBF, a metoda nu-SVR. Wobec tego mamy wpływ na parametr gamma, który występuje w metodzie nu-SVR. Domyślnie jego wartość to 1/ILOŚĆ_DANYCH (u nas 1/209). Parametr można przekazać do programu przez opcję '-gamma':

```
python app.py -i shuffled.data -gamma 0.1
```

Na początku programu możemy otrzymywać warning o treści:
```
RuntimeWarning: The _posixsubprocess module is not being used. Child process reliability may suffer if your program uses threads.
  "program uses threads.", RuntimeWarning)
```

Aby ten komunikat nas nie denerwował, można wykonać skrypt pythona z opcją ignorowania ostrzeżeń '-W ignore':

```
python -W ignore app.py -i shuffled.data -gamma 0.01
```