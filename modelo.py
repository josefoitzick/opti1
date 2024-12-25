import pulp as pl
import json
import os
import time

# Función para resolver una única instancia
def resolver_modelo(instancia, output_file):
    tiempo_inicio = time.time()

    # Leer parámetros del modelo
    dias = instancia["dias"]
    turnos = instancia["turnos"]
    categorias = instancia["categorias"]
    actividades = instancia["actividades"]
    demanda = instancia["demanda"]
    tiempo_atencion = instancia["tiempo_atencion"]
    min_personal = instancia["min_personal"]
    duracion_turno = instancia["duracion_turno"]

    # Crear el problema de optimización
    model = pl.LpProblem("Optimización_Urgencias_HLCM", pl.LpMinimize)

    # Variables de decisión
    medicos = pl.LpVariable.dicts(
        "Medicos",
        ((dia, turno, actividad) for dia in dias for turno in turnos for actividad in actividades),
        lowBound=0,
        cat="Integer",
    )

    horas_trabajadas = pl.LpVariable.dicts(
        "HorasTrabajadas",
        ((dia, turno, actividad) for dia in dias for turno in turnos for actividad in actividades),
        lowBound=0,
        cat="Continuous",
    )

    # Función objetivo: Minimizar el número total de médicos
    model += pl.lpSum(medicos[(dia, turno, actividad)] for dia in dias for turno in turnos for actividad in actividades), "Minimizar_Medicos"

    # Restricciones
    # 1. Satisfacer la demanda de atención diaria por categoría
    for dia in dias:
        for turno in turnos:
            for categoria in categorias:
                if categoria in tiempo_atencion:
                    model += (
                        pl.lpSum(horas_trabajadas[(dia, turno, actividad)] for actividad in actividades) 
                        >= demanda[dia][str(turno)][categoria] * tiempo_atencion[categoria],
                        f"Satisfacer_Demanda_{dia}_{turno}_{categoria}",
                    )

    # 2. Respetar el mínimo de trabajadores
    for dia in dias:
        for turno in turnos:
            for actividad in actividades:
                model += (
                    medicos[(dia, turno, actividad)] >= min_personal[actividad],
                    f"MinPersonal_{dia}_{turno}_{actividad}",
                )

    # 3. Horas trabajadas por turno no pueden exceder la duración del turno
    for dia in dias:
        for turno in turnos:
            for actividad in actividades:
                model += (
                    horas_trabajadas[(dia, turno, actividad)] <= duracion_turno * medicos[(dia, turno, actividad)],
                    f"MaxHoras_{dia}_{turno}_{actividad}",
                )

    # Resolver el problema
    solver = pl.CPLEX_CMD(msg=True) if pl.CPLEX().available() else pl.PULP_CBC_CMD(msg=True)
    status = model.solve(solver)

    tiempo_total = time.time() - tiempo_inicio
    valor_z = pl.value(model.objective) if model.status == pl.LpStatusOptimal else None

    # Guardar resultados
    with open(output_file, "a", encoding='utf-8') as f:
        f.write(f"\nInstancia: {instancia}\n")
        f.write(f"Estado: {pl.LpStatus[model.status]}\n")
        f.write(f"Tiempo de ejecución: {tiempo_total:.2f} segundos\n")
        
        if model.status == pl.LpStatusOptimal:
            f.write(f"Función objetivo (Z): {valor_z:.2f}\n")
            f.write("Asignación de médicos:\n")
            for dia in dias:
                for turno in turnos:
                    for actividad in actividades:
                        f.write(f"{dia}, Turno {turno}, {actividad}: {pl.value(medicos[(dia, turno, actividad)])} médicos\n")
        else:
            f.write("  No se encontró solución óptima.\n")
    
    return tiempo_total, valor_z

# Función para procesar múltiples instancias
def procesar_instancias(json_folder, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)
        
    tiempos = []
    valores_z = []
    instancias = []

    for filename in sorted(os.listdir(json_folder)):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(json_folder, filename), "r") as f:
                    instancia = json.load(f)
                print(f"Procesando {filename}...")
                tiempo, valor_z = resolver_modelo(instancia, output_file)
                tiempos.append(tiempo)
                valores_z.append(valor_z if valor_z is not None else 0)
                instancias.append(filename)
            except Exception as e:
                print(f"Error en {filename}: {str(e)}")
                tiempos.append(0)
                valores_z.append(0)
                instancias.append(filename)

    # Imprimir resumen
    print("\nResultados:")
    for i, instancia in enumerate(instancias):
        print(f"Instancia: {instancia}, Tiempo: {tiempos[i]:.2f}s, Valor Z: {valores_z[i]}")

    return tiempos, valores_z, instancias

# Ejecución del código principal
if __name__ == "__main__":
    json_folder = "instancias_json"
    output_file = "resultados.txt"

    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    procesar_instancias(json_folder, output_file)
