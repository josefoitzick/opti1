import pulp as pl
import json
import os
import time
import matplotlib.pyplot as plt

def resolver_problema(instancia, output_file):
    tiempo_inicio = time.time()

    # Cargar datos desde la instancia
    dias = instancia["dias"]
    turnos = instancia["turnos"]
    categorias = instancia["categorias"]
    actividades = instancia["actividades"]
    demanda = instancia["demanda"]
    tiempo_atencion = instancia["tiempo_atencion"]
    min_personal = instancia["min_personal"]
    duracion_turno = instancia["duracion_turno"]

    # Crear el modelo de optimización
    model = pl.LpProblem("Asignacion_Medicos", pl.LpMinimize)

    # Variables de decisión
    x = { 
        (actividad, dia, turno): pl.LpVariable(
            f"x_{actividad}_{dia}_{turno}", lowBound=0, cat=pl.LpInteger
        ) 
        for actividad in actividades for dia in dias for turno in turnos
    }

    # Función objetivo: Minimizar el número total de médicos
    model += pl.lpSum(x[actividad, dia, turno] for actividad in actividades for dia in dias for turno in turnos)

    # Restricciones
    # 1. Satisfacer la demanda de pacientes
    for dia in dias:
        for turno in turnos:
            for categoria in categorias:
                tiempo_requerido = tiempo_atencion[categoria]
                total_pacientes = sum(demanda[dia][str(turno)][categoria] for dia in dias)
                model += pl.lpSum(x[actividad, dia, turno] * duracion_turno 
                                  for actividad in actividades) >= total_pacientes * tiempo_requerido

    # 2. Respetar el mínimo de médicos por actividad y turno
    for actividad in actividades:
        for dia in dias:
            for turno in turnos:
                model += x[actividad, dia, turno] >= min_personal[actividad]

    # Resolver el problema
    solver = pl.CPLEX_CMD(msg=False) if pl.CPLEX().available() else pl.PULP_CBC_CMD(msg=False)
    status = model.solve(solver)

    tiempo_total = time.time() - tiempo_inicio
    valor_z = pl.value(model.objective) if model.status == pl.LpStatusOptimal else None

    # Guardar resultados
    with open(output_file, "a", encoding='utf-8') as f:
        f.write(f"\nInstancia: Días={len(dias)}, Turnos={len(turnos)}, Categorías={len(categorias)}\n")
        f.write(f"Estado: {pl.LpStatus[model.status]}\n")
        f.write(f"Tiempo de ejecución: {tiempo_total:.2f} segundos\n")

        if model.status == pl.LpStatusOptimal:
            f.write(f"Función objetivo (Z): {valor_z:.2f}\n")
            f.write("Asignación de médicos:\n")
            for actividad in actividades:
                for dia in dias:
                    for turno in turnos:
                        num_medicos = pl.value(x[actividad, dia, turno])
                        if num_medicos > 0:
                            f.write(f"  Actividad={actividad}, Día={dia}, Turno={turno}: {int(num_medicos)} médicos\n")
        else:
            f.write("  No se encontró solución óptima.\n")

    return tiempo_total, valor_z

def procesar_instancias(json_folder, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    tiempos = []
    valores_z = []

    for filename in sorted(os.listdir(json_folder)):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(json_folder, filename), "r") as f:
                    instancia = json.load(f)
                print(f"Procesando {filename}...")
                tiempo, valor_z = resolver_problema(instancia, output_file)
                tiempos.append(tiempo)
                valores_z.append(valor_z if valor_z is not None else 0)
            except Exception as e:
                print(f"Error en {filename}: {str(e)}")
                tiempos.append(0)
                valores_z.append(0)

    # Crear gráfico de resultados
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(tiempos)), tiempos, label='Tiempo de ejecución')
    plt.plot(range(len(valores_z)), valores_z, label='Valor de la función objetivo (Z)')
    plt.xlabel('Instancia')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid()
    plt.savefig('resultados.png')
    print("Gráfico generado: 'resultados.png'")

# Configuración y ejecución
json_folder = "instancias_json"
output_file = "resumen_resultados.txt"

if not os.path.exists(json_folder):
    os.makedirs(json_folder)
procesar_instancias(json_folder, output_file)
