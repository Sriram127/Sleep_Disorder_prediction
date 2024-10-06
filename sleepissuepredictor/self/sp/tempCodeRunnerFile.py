return render_template('predict.html', prediction=prediction)
    else:
        return render_template('index.html')